use std::{sync::Arc, sync::RwLock, sync::atomic::{AtomicU64, Ordering}, time::Instant};

use colored::*;
use drillx::{
    equix::{self},
    Hash, Solution,
};
use ore_api::{
    consts::{BUS_ADDRESSES, BUS_COUNT, EPOCH_DURATION},
    state::{Bus, Config, Proof},
};
use ore_utils::AccountDeserialize;
use rand::Rng;
use solana_program::pubkey::Pubkey;
use solana_rpc_client::spinner;
use solana_sdk::signer::Signer;
use reqwest::Client;
use serde_json::json;
use chrono::Utc;
use std::cell::RefCell;
use futures::stream::{self, StreamExt};

use crate::{
    args::MineArgs,
    send_and_confirm::ComputeBudget,
    utils::{
        amount_u64_to_string, get_clock, get_config, get_proof_with_authority, proof_pubkey,
    },
    Miner,
};

const DISCORD_WEBHOOK_URL: &str = "xxxxxxxxxxxxxxx"; // Replace with your Discord webhook URL

impl Miner {
    pub async fn mine(&self, args: MineArgs) {
        // Open account, if needed.
        let signer = self.signer();
        self.open().await;

        // Check num threads
        self.check_num_cores(args.cores);

        // Start mining loop
        let mut last_hash_at = 0;
        let mut last_balance = 0;
        loop {
            // Fetch proof
            let config = get_config(&self.rpc_client).await;
            let proof =
            get_proof_with_authority(&self.rpc_client, signer.pubkey())
                    .await;
            println!(
                "\n\nStake: {} ORE\n{}  Multiplier: {:12}x",
                amount_u64_to_string(proof.balance),
                if last_hash_at.gt(&0) {
                    format!(
                        "  Change: {} ORE\n",
                        amount_u64_to_string(proof.balance.saturating_sub(last_balance))
                    )
                } else {
                    "".to_string()
                },
                calculate_multiplier(proof.balance, config.top_balance)
            );
            last_hash_at = proof.last_hash_at;
            last_balance = proof.balance;

            // Calculate cutoff time
            let cutoff_time = self.get_cutoff(proof, args.buffer_time).await;

            // Run drillx
            let solution =
                Self::find_hash_par(proof, cutoff_time, config.min_difficulty as u32)
                    .await;

            // Build instruction set
            let mut ixs = vec![ore_api::instruction::auth(proof_pubkey(signer.pubkey()))];
            let mut compute_budget = 500_000;
            if self.should_reset(config).await && rand::thread_rng().gen_range(0..100).eq(&0) {
                compute_budget += 100_000;
                ixs.push(ore_api::instruction::reset(signer.pubkey()));
            }

            // Build mine ix
            ixs.push(ore_api::instruction::mine(
                signer.pubkey(),
                signer.pubkey(),
                self.find_bus().await,
                solution,
            ));

            // Submit transaction
            match self.send_and_confirm(&ixs, ComputeBudget::Fixed(compute_budget), false).await {
                Ok(tx_hash) => {
                    println!("{}", "Transaction confirmed successfully.".bold().green());
                    // Send Discord notification with transaction hash
                    self.send_discord_webhook(&solution, &tx_hash.to_string(), proof.balance, signer.pubkey().to_string()).await;
                },
                Err(err) => {
                    println!("{}: {}", "ERROR".bold().red(), err);
                }
            }
        }
    }

    async fn find_hash_par(
        proof: Proof,
        cutoff_time: u64,
        min_difficulty: u32,
    ) -> Solution {
        // Shared state for tracking the best difficulty and total hashes
        let global_best_difficulty = Arc::new(RwLock::new(0u32));
        let global_total_hashes = Arc::new(AtomicU64::new(0));
        let start_time = Instant::now();
        let num_threads = num_cpus::get(); // Get the number of logical cores

        // Adaptive nonce range based on thread performance
        let nonce_step = u64::MAX / num_threads as u64;
        let nonce_ranges: Vec<(u64, u64)> = (0..num_threads)
            .map(|i| (nonce_step * i as u64, nonce_step * (i as u64 + 1)))
            .collect();

        // Thread-local storage for SolverMemory
        thread_local! {
            static SOLVER_MEMORY: RefCell<equix::SolverMemory> = RefCell::new(equix::SolverMemory::new());
        }

        // Spawn threads for each logical core
        let handles: Vec<_> = nonce_ranges.into_iter()
            .enumerate()
            .map(|(i, (start_nonce, end_nonce))| {
                let global_total_hashes = Arc::clone(&global_total_hashes);
                let global_best_difficulty = Arc::clone(&global_best_difficulty);
                std::thread::spawn({
                    let proof = proof.clone();
                    move || {
                        // Pin to core
                        let core_ids = core_affinity::get_core_ids().unwrap();
                        if let Some(core_id) = core_ids.get(i) {
                            let _ = core_affinity::set_for_current(*core_id);
                        }

                        // Start hashing
                        let mut nonce = start_nonce;
                        let mut best_nonce = nonce;
                        let mut best_difficulty = 0;
                        let mut best_hash = Hash::default();
                        SOLVER_MEMORY.with(|memory| {
                            let mut memory = memory.borrow_mut();
                            while nonce < end_nonce {
                                // Create hash
                                if let Ok(hx) = drillx::hash_with_memory(
                                    &mut memory,
                                    &proof.challenge,
                                    &nonce.to_le_bytes(),
                                ) {
                                    let difficulty = hx.difficulty();
                                    if difficulty > best_difficulty {
                                        best_nonce = nonce;
                                        best_difficulty = difficulty;
                                        best_hash = hx;
                                        // Update global best difficulty
                                        let mut best_diff_lock = global_best_difficulty.write().unwrap();
                                        if best_difficulty > *best_diff_lock {
                                            *best_diff_lock = best_difficulty;
                                        }
                                    }
                                }
                                global_total_hashes.fetch_add(1, Ordering::Relaxed);

                                // Exit if time has elapsed
                                if start_time.elapsed().as_secs() >= cutoff_time {
                                    let best_diff_lock = global_best_difficulty.read().unwrap();
                                    if *best_diff_lock >= min_difficulty {
                                        break;
                                    }
                                }

                                // Increment nonce
                                nonce += 1;
                            }
                        });

                        // Return the best nonce
                        (best_nonce, best_difficulty, best_hash)
                    }
                })
            })
            .collect();

        // Join handles and return the best nonce
        let mut best_nonce = 0;
        let mut best_difficulty = 0;
        let mut best_hash = Hash::default();
        for h in handles {
            if let Ok((nonce, difficulty, hash)) = h.join() {
                if difficulty > best_difficulty {
                    best_difficulty = difficulty;
                    best_nonce = nonce;
                    best_hash = hash;
                }
            }
        }

        let total_hashes = global_total_hashes.load(Ordering::Relaxed);
        let elapsed_time = start_time.elapsed().as_secs_f64();
        let hash_rate = total_hashes as f64 / elapsed_time;

        // Log the best hash and performance metrics
        println!(
            "Best hash: {} (difficulty {}, {:.2} H/s)",
            bs58::encode(best_hash.h).into_string(),
            best_difficulty,
            hash_rate,
        );

        Solution::new(best_hash.d, best_nonce.to_le_bytes())
    }

    pub fn check_num_cores(&self, cores: u64) {
        let num_cores = num_cpus::get() as u64;
        if cores.gt(&num_cores) {
            println!(
                "{} Cannot exceeds available cores ({})",
                "WARNING".bold().yellow(),
                num_cores
            );
        }
    }

    async fn should_reset(&self, config: Config) -> bool {
        if let Some(clock) = get_clock(&self.rpc_client).await {
            config
                .last_reset_at
                .saturating_add(EPOCH_DURATION)
                .saturating_sub(5) // Buffer
                .le(&clock.unix_timestamp)
        } else {
            false
        }
    }

    async fn get_cutoff(&self, proof: Proof, buffer_time: u64) -> u64 {
        if let Some(clock) = get_clock(&self.rpc_client).await {
            proof
                .last_hash_at
                .saturating_add(60)
                .saturating_sub(buffer_time as i64)
                .saturating_sub(clock.unix_timestamp)
                .max(0) as u64
        } else {
            60 // Default cutoff time if clock retrieval fails
        }
    }

    async fn find_bus(&self) -> Pubkey {
        // Fetch the bus with the largest balance
        if let Ok(accounts) = self.rpc_client.get_multiple_accounts(&BUS_ADDRESSES).await {
            let top_bus = Arc::new(RwLock::new((0u64, BUS_ADDRESSES[0])));

            // Process accounts in parallel
            stream::iter(accounts)
                .for_each_concurrent(None, |account| {
                    let top_bus = Arc::clone(&top_bus);
                    async move {
                        if let Some(account) = account {
                            if let Ok(bus) = Bus::try_from_bytes(&account.data) {
                                let mut top_bus_lock = top_bus.write().unwrap();
                                if bus.rewards > top_bus_lock.0 {
                                    *top_bus_lock = (bus.rewards, BUS_ADDRESSES[bus.id as usize]);
                                }
                            }
                        }
                    }
                })
                .await;

            let top_bus_lock = top_bus.read().unwrap();
            return top_bus_lock.1;
        }

        // Otherwise return a random bus
        let i = rand::thread_rng().gen_range(0..BUS_COUNT);
        BUS_ADDRESSES[i]
    }

    async fn send_discord_webhook(&self, solution: &Solution, tx_hash: &str, stake_balance: u64, wallet_address: String) {
        let client = Client::new();
        let diff = solution.to_hash().difficulty();
        let timestamp = Utc::now().to_rfc3339();
        let tx_url = format!("https://solscan.io/tx/{}", tx_hash);
        let stake_balance_str = amount_u64_to_string(stake_balance); 

        // Create the embed payload for the Discord webhook
        let embed = json!({
            "title": "Mining Successful!",
            "color": 65280, // Green color
            "description": format!("Wallet: {}", wallet_address),
            "fields": [
                {
                    "name": "Stake Balance",
                    "value": format!("{} ORE", stake_balance_str),
                    "inline": true
                },
                {
                    "name": "Difficulty",
                    "value": format!("{:?}", diff),
                    "inline": true
                },
                {
                    "name": "Details",
                    "value": format!("[Solscan]({})", tx_url),
                    "inline": true
                }
            ],
            "timestamp": timestamp
        });

        // Add a ping if difficulty is 30 or higher
        let mut content = String::new();
        if diff >= 30 {
            content = "@everyone".to_string();
        }

        let payload = json!({
            "content": content,
            "embeds": [embed]
        });

        // Send the payload to the Discord webhook URL
        match client.post(DISCORD_WEBHOOK_URL).json(&payload).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    println!("{}", "Discord notification sent successfully.".bold().green());
                } else {
                    println!(
                        "{}: Failed to send Discord notification. Status: {}",
                        "ERROR".bold().red(),
                        response.status()
                    );
                }
            },
            Err(err) => {
                println!(
                    "{}: Failed to send Discord notification. Error: {}",
                    "ERROR".bold().red(),
                    err
                );
            }
        }
    }
}

fn calculate_multiplier(balance: u64, top_balance: u64) -> f64 {
    1.0 + (balance as f64 / top_balance as f64).min(1.0f64)
}

fn format_duration(seconds: u32) -> String {
    let minutes = seconds / 60;
    let remaining_seconds = seconds % 60;
    format!("{:02}:{:02}", minutes, remaining_seconds)
}

#!/usr/bin/env python3
import pandas as pd


def cost_analysis():
        LOG_FILE = f"logs/usage/2025-04-16T16:27:32.475138-api_usage_log.csv"
        df = pd.read_csv(LOG_FILE)
        print(f"Total user tokens: {df['user_tokens'].sum():,}")
        print(f"Total context tokens: {df['system_tokens'].sum():,}")
        print(f"Total input tokens (estimation): {df['est_input_token'].sum():,}")
        print(f"Total input tokens: {df['input_tokens'].sum():,}")
        print(f"Total generated tokens: {df['output_tokens'].sum():,}")
        print(f"Estimated cost: ${df['estimated_cost'].sum():.4f}")



if __name__ == "__main__":
        cost_analysis()
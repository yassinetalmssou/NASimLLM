import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(r'C:\Users\yassi\Documents\VUB\Master\2MA\Thesis\NASimLLM')
BASE_DIR = PROJECT_ROOT / 'runs' / 'rq3b'
CONDITIONS = ['full', 'no_history', 'no_avoidlist', 'verbose_prompt', 'llm_cached']

print('='*80)
print('RQ3b ABLATION RESULTS ANALYSIS')
print('='*80)

results = {}
for condition in CONDITIONS:
    csv_path = BASE_DIR / condition / 'seed0' / 'train.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        last_10 = df.tail(10)
        
        results[condition] = {
            'success_rate': last_10['success'].mean() * 100,
            'avg_reward': last_10['reward'].mean(),
            'avg_length': last_10['length'].mean(),
            'llm_time': last_10['t_llm_s'].mean(),
            'cache_hits': df['cache_hits'].iloc[-1],
            'cache_misses': df['cache_misses'].iloc[-1],
            'first_5_rewards': df.head(5)['reward'].tolist()
        }
        
        print(f'\n{condition.upper().replace("_", " ")}:')
        print(f'  Success Rate (last 10 eps): {results[condition]["success_rate"]:.1f}%')
        print(f'  Avg Reward (last 10 eps):   {results[condition]["avg_reward"]:.1f}')
        print(f'  Avg Length (last 10 eps):   {results[condition]["avg_length"]:.1f} steps')
        print(f'  LLM Time per episode:       {results[condition]["llm_time"]:.2f}s')
        print(f'  Cache hits/misses:          {results[condition]["cache_hits"]:.0f} / {results[condition]["cache_misses"]:.0f}')
        first_5 = [round(r, 1) for r in results[condition]['first_5_rewards']]
        print(f'  First 5 episode rewards:    {first_5}')

print('\n' + '='*80)
print('COMPARISON TO FULL SYSTEM (baseline):')
print('='*80)

if 'full' in results:
    baseline = results['full']
    for condition in ['no_history', 'no_avoidlist', 'verbose_prompt', 'llm_cached']:
        if condition in results:
            r = results[condition]
            success_diff = r['success_rate'] - baseline['success_rate']
            reward_diff = r['avg_reward'] - baseline['avg_reward']
            length_diff = r['avg_length'] - baseline['avg_length']
            time_diff = r['llm_time'] - baseline['llm_time']
            
            print(f'\n{condition.upper().replace("_", " ")}:')
            print(f'  Success Rate: {success_diff:+.1f} pp  ({"BETTER" if success_diff > 0 else "WORSE" if success_diff < 0 else "SAME"})')
            print(f'  Avg Reward:   {reward_diff:+.1f}     ({"BETTER" if reward_diff > 0 else "WORSE" if reward_diff < 0 else "SAME"})')
            print(f'  Avg Length:   {length_diff:+.1f}     ({"SHORTER" if length_diff < 0 else "LONGER" if length_diff > 0 else "SAME"})')
            print(f'  LLM Time:     {time_diff:+.2f}s    ({"FASTER" if time_diff < 0 else "SLOWER" if time_diff > 0 else "SAME"})')
            
            # Check if trajectories are identical
            identical = results[condition]['first_5_rewards'] == baseline['first_5_rewards']
            print(f'  Identical trajectories: {"YES ⚠️" if identical else "NO ✅"}')

print('\n' + '='*80)

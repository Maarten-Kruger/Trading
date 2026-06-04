import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and fill missing Bid/Ask
df = pd.read_csv('EURUSD_1.6_Tick Data.csv', sep='\t')
df['<BID>'] = df['<BID>'].ffill().bfill()
df['<ASK>'] = df['<ASK>'].ffill().bfill()

bid = df['<BID>'].values
ask = df['<ASK>'].values

rr_levels = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
r_sizes = [20, 50, 75, 100]
entry_indices = np.arange(0, len(bid), 100) # Every 100th tick

# Matrices to store EV and total trades
ev_matrix = np.zeros((len(r_sizes), len(rr_levels)))

for i_r, r_size in enumerate(r_sizes):
    for i_rr, rr in enumerate(rr_levels):
        r_value = r_size * 0.00001
        tp_value = r_value * rr
        
        wins, losses = 0, 0
        
        for i in entry_indices:
            spread = ask[i] - bid[i]
            
            # Only take the trade if spread is less than the stop loss size
            if spread < r_value:
                # UP trade logic
                entry_up = ask[i]
                sl_up = entry_up - r_value
                tp_up = entry_up + tp_value
                
                # DOWN trade logic
                entry_down = bid[i]
                sl_down = entry_down + r_value
                tp_down = entry_down - tp_value
                
                up_resolved, down_resolved = False, False
                
                # Scan forward to find the outcome
                for j in range(i+1, len(bid)):
                    # Evaluate UP trade (Exit at BID)
                    if not up_resolved:
                        if bid[j] <= sl_up:
                            losses += 1
                            up_resolved = True
                        elif bid[j] >= tp_up:
                            wins += 1
                            up_resolved = True
                            
                    # Evaluate DOWN trade (Exit at ASK)
                    if not down_resolved:
                        if ask[j] >= sl_down:
                            losses += 1
                            down_resolved = True
                        elif ask[j] <= tp_down:
                            wins += 1
                            down_resolved = True
                            
                    # Break early if both directions hit an exit
                    if up_resolved and down_resolved:
                        break
                        
        total_trades = wins + losses
        
        if total_trades > 0:
            win_rate = wins / total_trades
            loss_rate = losses / total_trades
            ev = (win_rate * rr) - (loss_rate * 1)
        else:
            ev = np.nan # No trades taken due to spread
            
        ev_matrix[i_r, i_rr] = ev

# Plotting the EV Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(ev_matrix, annot=True, cmap='RdYlGn', center=0, 
            xticklabels=rr_levels, yticklabels=r_sizes, fmt=".3f", 
            cbar_kws={'label': 'Expected Value (R)'})
plt.title('Expected Value (EV) Matrix of Random Trades\n(Filtered by Spread < R-Size)')
plt.xlabel('Risk/Reward (RR)')
plt.ylabel('R-Size (Points)')
plt.tight_layout()
plt.savefig('ev_matrix.png')
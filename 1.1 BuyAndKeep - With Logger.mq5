#property strict
#property script_show_inputs

#include <Trade/Trade.mqh>

//+------------------------------------------------------------------+
//| Expert Advisor: Buy and Keep                                     |
//| Version: 1.1                                                     |
//| Implements a simple mean reversion buy strategy that             |
//| accumulates positions when price drops by a threshold and        |
//| manages risk via equity drawdown control.                        |
//+------------------------------------------------------------------+

//---- trading inputs
input double   RiskPercent       = 1.0;    // percent of equity risked per trade
input int      BACKCHECK         = 5;      // candles back for reference high/low
input double   THRESHOLD         = 5.0;    // threshold percent (decimals like 0.01 = 0.01%; negative flips logic)
input double   TakeProfitPips    = 100;    // take profit distance in pips
input double   MaxDrawdownPct    = 30.0;   // maximum allowed drawdown before trimming positions
input long     MagicNumber       = 1101;   // magic number for trade identification
input bool     FridayCloseAll    = true;   // enable Friday cutoff risk management
input int      FridayCutoffHour  = 20;     // server hour after which trades are closed on Friday
input int      FridayCutoffMinute= 0;      // server minute after which trades are closed on Friday

//---- optimization weights (sum should equal 100)
input double   Wt = 33.0;               // weight for trade density
input double   Wp = 33.0;               // weight for monthly profit
input double   Wd = 34.0;               // weight for drawdown penalty

//---- global variables
CTrade         trade;                    // trading object for order execution
datetime       last_bar_time = 0;        // time of last processed bar
double         max_equity    = 0;        // tracks peak equity for drawdown calculation
int            total_bars    = 0;        // bars in test period
int            total_months  = 0;        // months in test period
datetime       last_server_time = 0;     // last reliable trade server time snapshot
datetime       last_local_snapshot = 0;  // matching local time when server time captured
string         csv_filename;             // output CSV file for closed trade logging
bool           csv_header_ready = false; // indicates header already written

//+------------------------------------------------------------------+
//| Helper forward declarations                                      |
//+------------------------------------------------------------------+
bool   AllowNewTrades();
void   FridayRiskManagement();
bool   GetCurrentTradingTime(MqlDateTime &out);
ulong  FindWorstLosingTicket();
void   CloseAllPositionsByMagic();
int    CountPositionsByMagic();
void   EnsureCsvHeader();
void   LogClosedTrade(const MqlTradeTransaction &trans);
datetime ExtractOpenTime(ulong position_id);
void   AppendTradeCsv(ulong ticket, datetime open_time, datetime close_time, double equity);

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   max_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   trade.SetExpertMagicNumber(MagicNumber);
   EventSetTimer(60);

   //--- prepare CSV logging infrastructure
   csv_filename = StringFormat("%s_BuyAndKeep_Trades.csv", _Symbol);
   EnsureCsvHeader();
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Tick handler                                                     |
//+------------------------------------------------------------------+
void OnTick()
  {
   //--- manage equity drawdown every tick
   ManageDrawdown();

   //--- handle Friday cutoff logic
   FridayRiskManagement();

   //--- new bar detection
   datetime current = iTime(_Symbol,_Period,0);
   if(current != last_bar_time)
     {
      last_bar_time = current;
      OnNewBar();
     }
  }

//+------------------------------------------------------------------+
//| Timer handler to enforce Friday protections without ticks         |
//+------------------------------------------------------------------+
void OnTimer()
  {
   FridayRiskManagement();
  }

//+------------------------------------------------------------------+
//| Trade transaction handler to capture closed positions            |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
   //--- only interested in completed closing deals belonging to this EA
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD)
      return;
   if(trans.deal_entry != DEAL_ENTRY_OUT)
      return;
   if((long)trans.magic != MagicNumber)
      return;

   ulong position_id = trans.position_id;
   if(position_id == 0)
      position_id = trans.position;

   //--- skip if position still exists (partial close)
   if(position_id != 0 && PositionSelectByTicket(position_id))
      return;

   LogClosedTrade(trans);
  }

//+------------------------------------------------------------------+
//| Called on formation of a new bar                                 |
//+------------------------------------------------------------------+
void OnNewBar()
  {
   //--- ensure enough history is available
   if(Bars(_Symbol,_Period) <= BACKCHECK)
      return;

   //--- obey Friday trade cutoff rules
   if(!AllowNewTrades())
      return;

   double current_low = iLow(_Symbol,_Period,0);

   double threshold_normalized = THRESHOLD;

   if(threshold_normalized >= 0.0)
     {
      //--- mean reversion logic: buy on drop from reference high
      double reference_high = iHigh(_Symbol,_Period,BACKCHECK);
      if(reference_high <= 0.0)
         return;
      double drop_percent = (reference_high - current_low) / reference_high * 100.0;

      if(drop_percent > threshold_normalized)
         OpenBuy();
     }
   else
     {
      //--- momentum logic: buy on strength from reference low
      double reference_low = iLow(_Symbol,_Period,BACKCHECK);
      if(reference_low <= 0.0)
         return;
      double gain_percent = (current_low - reference_low) / reference_low * 100.0;

      if(gain_percent > MathAbs(threshold_normalized))
         OpenBuy();
     }
  }

//+------------------------------------------------------------------+
//| Opens a buy order with TP only                                   |
//+------------------------------------------------------------------+
void OpenBuy()
  {
   double ask = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   double tp  = NormalizeDouble(ask + TakeProfitPips * _Point, _Digits);

   //--- calculate position size based on equity percentage
   double volume = CalculateVolume();
   if(volume <= 0)
      return;

   trade.Buy(volume,NULL,ask,0,tp);
  }

//+------------------------------------------------------------------+
//| Calculates trade volume from equity percentage                   |
//| Approximates risk by equating take profit value to risk amount   |
//+------------------------------------------------------------------+
double CalculateVolume()
  {
   double equity    = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_money= equity * RiskPercent / 100.0;
   double tick_value= SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE);
   double lot_step  = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   double min_lot   = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   double volume    = risk_money / (tick_value * TakeProfitPips / _Point);
   volume           = MathMax(min_lot, MathFloor(volume/lot_step)*lot_step);

   //--- derive volume precision from step size
   int digits = (int)MathRound(-MathLog10(lot_step));
   return NormalizeDouble(volume,digits);
  }

//+------------------------------------------------------------------+
//| Monitors drawdown and closes worst positions if necessary        |
//+------------------------------------------------------------------+
void ManageDrawdown()
  {
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity > max_equity)
      max_equity = equity;
   if(max_equity <= 0.0)
      return;

   double drawdown = (max_equity - equity) / max_equity * 100.0;
   bool   trimmed        = false;
   bool   attempted_trim = false;

   //--- close worst losing positions until drawdown is within limit
   while(drawdown > MaxDrawdownPct)
     {
      ulong ticket = FindWorstLosingTicket();
      if(ticket == 0)
        {
         attempted_trim = true;
         break;
        }

      attempted_trim = true;

      if(!trade.PositionClose(ticket))
         break;

      trimmed = true;

      trimmed = true;

      equity = AccountInfoDouble(ACCOUNT_EQUITY);
      if(max_equity <= 0.0)
         break;
      drawdown = (max_equity - equity) / max_equity * 100.0;
     }

   int open_positions = CountPositionsByMagic();

   if(trimmed)
     {
      //--- after forced liquidation, treat the current balance as the new peak
      //    so the EA can resume trading instead of repeatedly closing
      if(open_positions == 0)
        {
         max_equity = equity;
        }
     }
   else if(attempted_trim && open_positions == 0 && equity > 0.0)
     {
      //--- drawdown remains but there are no positions to close; reset baseline
      max_equity = equity;
     }
   else if(drawdown > 0.0 && open_positions == 0 && equity > 0.0)
     {
      //--- realized loss with no exposure should not block new entries
      max_equity = equity;
     }
  }

//| Finds the worst losing position ticket for this EA                |
//+------------------------------------------------------------------+
ulong FindWorstLosingTicket()
  {
   bool  found        = false;
   ulong worst_ticket = 0;
   double worst_profit= 0.0;

   for(int i=PositionsTotal()-1;i>=0;i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != MagicNumber)
         continue;

      double profit = PositionGetDouble(POSITION_PROFIT);
      if(!found || profit < worst_profit)
        {
         found         = true;
         worst_ticket  = ticket;
         worst_profit  = profit;
        }
     }

   if(!found || worst_profit >= 0.0)
      return(0);

   return(worst_ticket);
  }

//+------------------------------------------------------------------+
//| Determines whether new trades are allowed under Friday rules      |
//+------------------------------------------------------------------+
bool AllowNewTrades()
  {
   if(!FridayCloseAll)
      return(true);

   MqlDateTime t;
   if(!GetCurrentTradingTime(t))
      return(true);
   if(t.day_of_week != 5) // not Friday
      return(true);

   if(t.hour > FridayCutoffHour)
      return(false);
   if(t.hour == FridayCutoffHour && t.min >= FridayCutoffMinute)
      return(false);

   return(true);
  }

//+------------------------------------------------------------------+
//| Applies Friday close logic: block entries and close positions     |
//+------------------------------------------------------------------+
void FridayRiskManagement()
  {
   if(!FridayCloseAll)
      return;

   MqlDateTime t;
   if(!GetCurrentTradingTime(t))
      return;

   if(t.day_of_week != 5)
      return;

   //--- close all positions after configured time
   bool should_close = false;
   if(t.hour > FridayCutoffHour)
      should_close = true;
   else if(t.hour == FridayCutoffHour && t.min >= FridayCutoffMinute)
      should_close = true;

   if(should_close)
      CloseAllPositionsByMagic();
  }

//+------------------------------------------------------------------+
//| Retrieves current trading time (strategy tester or live server)   |
//+------------------------------------------------------------------+
bool GetCurrentTradingTime(MqlDateTime &out)
  {
   datetime server_time = TimeCurrent();
   datetime local_time  = TimeLocal();

   //--- update the server snapshot whenever a newer timestamp arrives
   if(server_time > 0 && (last_server_time == 0 || server_time >= last_server_time))
     {
      last_server_time    = server_time;
      last_local_snapshot = local_time;
     }
   else if(last_server_time > 0 && local_time > last_local_snapshot)
     {
      //--- extrapolate forward using local clock when ticks stop updating server time
      long delta = (long)(local_time - last_local_snapshot);
      last_server_time    += (datetime)delta;
      last_local_snapshot  = local_time;

     }

   if(last_server_time <= 0)
      return(false);

   TimeToStruct(last_server_time,out);
   return(true);
  }


//+------------------------------------------------------------------+
//| Closes every open position with the EA's magic number             |
//+------------------------------------------------------------------+
void CloseAllPositionsByMagic()
  {
   for(int i=PositionsTotal()-1;i>=0;i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != MagicNumber)
         continue;
      trade.PositionClose(ticket);
     }
  }

//+------------------------------------------------------------------+
//| Counts open positions controlled by this EA                       |
//+------------------------------------------------------------------+
int CountPositionsByMagic()
  {
   int count = 0;
   for(int i=PositionsTotal()-1;i>=0;i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != MagicNumber)
         continue;
      count++;
     }
   return(count);
  }

//+------------------------------------------------------------------+
//| Ensures CSV file exists with a header                            |
//+------------------------------------------------------------------+
void EnsureCsvHeader()
  {
   if(csv_header_ready)
      return;

   if(FileIsExist(csv_filename,FILE_COMMON))
     {
      csv_header_ready = true;
      return;
     }

   int handle = FileOpen(csv_filename,
                         FILE_WRITE|FILE_CSV|FILE_COMMON|FILE_SHARE_READ|FILE_SHARE_WRITE,
                         ',');
   if(handle == INVALID_HANDLE)
      return;

   FileWrite(handle,"Ticket","Open Time","Close Time","Equity");
   FileClose(handle);
   csv_header_ready = true;
  }

//+------------------------------------------------------------------+
//| Extracts the initial open time for a closed position              |
//+------------------------------------------------------------------+
datetime ExtractOpenTime(ulong position_id)
  {
   if(position_id == 0)
      return(0);

   if(!HistorySelectByPosition(position_id))
      return(0);

   int deals = (int)HistoryDealsTotal();
   for(int i=0; i<deals; i++)
     {
      ulong deal_ticket = HistoryDealGetTicket(i);
      if(deal_ticket == 0)
         continue;

      long entry = HistoryDealGetInteger(deal_ticket,DEAL_ENTRY);
      if(entry == DEAL_ENTRY_IN)
         return((datetime)HistoryDealGetInteger(deal_ticket,DEAL_TIME));
     }

   return(0);
  }

//+------------------------------------------------------------------+
//| Appends closed trade information to CSV                           |
//+------------------------------------------------------------------+
void AppendTradeCsv(ulong ticket, datetime open_time, datetime close_time, double equity)
  {
   int handle = FileOpen(csv_filename,
                         FILE_READ|FILE_WRITE|FILE_CSV|FILE_COMMON|FILE_SHARE_READ|FILE_SHARE_WRITE,
                         ',');
   if(handle == INVALID_HANDLE)
      return;

   FileSeek(handle,0,SEEK_END);

   string open_str  = open_time  > 0 ? TimeToString(open_time, TIME_DATE|TIME_SECONDS)   : "";
   string close_str = close_time > 0 ? TimeToString(close_time,TIME_DATE|TIME_SECONDS) : "";

   FileWrite(handle,
             ticket,
             open_str,
             close_str,
             DoubleToString(equity,2));
   FileClose(handle);
  }

//+------------------------------------------------------------------+
//| Logs a closed trade: print message and persist to CSV             |
//+------------------------------------------------------------------+
void LogClosedTrade(const MqlTradeTransaction &trans)
  {
   ulong ticket = 0;
   if(trans.position_id != 0)
      ticket = trans.position_id;
   else if(trans.position != 0)
      ticket = trans.position;
   else if(trans.deal != 0)
      ticket = trans.deal;

   ulong reference_id  = trans.position_id != 0 ? trans.position_id : trans.position;
   datetime open_time  = ExtractOpenTime(reference_id);
   datetime close_time = (datetime)trans.time;
   double   equity     = AccountInfoDouble(ACCOUNT_EQUITY);

   EnsureCsvHeader();
   AppendTradeCsv(ticket,open_time,close_time,equity);

   PrintFormat("Trade %I64u closed at %s. Equity: %.2f",
               ticket,
               close_time > 0 ? TimeToString(close_time,TIME_DATE|TIME_SECONDS) : "unknown",
               equity);
  }

//+------------------------------------------------------------------+
//| Calculates the number of bars and months in the tested period    |
//+------------------------------------------------------------------+
void CalculateBarsAndMonths()
  {
   //--- total number of bars in history
   total_bars = Bars(_Symbol,_Period);
   if(total_bars<=0)
     {
      total_months = 0;
      return;
     }

   //--- times of first and last bar for month calculation
   datetime first = iTime(_Symbol,_Period,total_bars-1);
   datetime last  = iTime(_Symbol,_Period,0);

   MqlDateTime s1,s2;
   TimeToStruct(first,s1);
   TimeToStruct(last,s2);
   total_months = (s2.year - s1.year) * 12 + (s2.mon - s1.mon) + 1;
  }

//+------------------------------------------------------------------+
//| Custom optimization criterion                                    |
//+------------------------------------------------------------------+
double OnTester()
  {
   //--- compute span statistics
   CalculateBarsAndMonths();

   double trades       = TesterStatistics(STAT_TRADES);
   double total_profit = TesterStatistics(STAT_PROFIT);
   double start_equity = TesterStatistics(STAT_INITIAL_DEPOSIT);
   double drawdown     = TesterStatistics(STAT_EQUITY_DDREL_PERCENT) / 100.0;

   double T = 0.0;
   double P = 0.0;
   if(total_bars   > 0) T = trades / (double)total_bars;
   if(total_months > 0) P = (total_profit / start_equity) / (double)total_months;

   double wt = Wt/100.0;
   double wp = Wp/100.0;
   double wd = Wd/100.0;

   double score = (T*wt + P*wp - drawdown*wd) * 100.0;
   return score;
  }

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
  }

//+------------------------------------------------------------------+

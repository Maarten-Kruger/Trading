#property strict
#property script_show_inputs

#include <Trade/Trade.mqh>

//+------------------------------------------------------------------+
//| Expert Advisor: Buy and Keep                                     |
//| Version: 1.0                                                     |
//| Implements a simple mean reversion buy strategy that             |
//| accumulates positions when price drops by a threshold and        |
//| manages risk via equity drawdown control.                        |
//+------------------------------------------------------------------+

//---- trading inputs
input double   RiskPercent     = 1.0;    // percent of equity risked per trade
input int      BACKCHECK       = 5;      // candles back for reference high
input double   THRESHOLD       = 5.0;    // percent drop from reference high to trigger buy
input double   TakeProfitPips  = 100;    // take profit distance in pips
input double   MaxDrawdownPct  = 30.0;   // maximum allowed drawdown before trimming positions

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

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   max_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Tick handler                                                     |
//+------------------------------------------------------------------+
void OnTick()
  {
   //--- manage equity drawdown every tick
   ManageDrawdown();

   //--- new bar detection
   datetime current = iTime(_Symbol,_Period,0);
   if(current != last_bar_time)
     {
      last_bar_time = current;
      OnNewBar();
     }
  }

//+------------------------------------------------------------------+
//| Called on formation of a new bar                                 |
//+------------------------------------------------------------------+
void OnNewBar()
  {
   //--- ensure enough history is available
   if(Bars(_Symbol,_Period) <= BACKCHECK)
      return;

   //--- calculate percentage drop from high BACKCHECK bars ago
   double reference_high = iHigh(_Symbol,_Period,BACKCHECK);
   double current_low    = iLow(_Symbol,_Period,0);
   if(reference_high <= 0.0)
      return;
   double drop_percent = (reference_high - current_low) / reference_high * 100.0;

   //--- entry condition: price has fallen more than THRESHOLD percent
   if(drop_percent > THRESHOLD)
      OpenBuy();
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
   int    digits    = (int)SymbolInfoInteger(_Symbol,SYMBOL_VOLUME_DIGITS);
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
   double drawdown = (max_equity - equity) / max_equity * 100.0;
   
   //--- close worst losing positions until drawdown is within limit
   while(drawdown > MaxDrawdownPct && PositionsTotal() > 0)
     {
      int    index_worst = -1;
      double profit_worst = 0.0;
      for(int i=0;i<PositionsTotal();i++)
        {
         if(PositionGetTicket(i) == 0)
            continue;
         double profit = PositionGetDouble(POSITION_PROFIT);
         if(index_worst == -1 || profit < profit_worst)
           {
            index_worst   = i;
            profit_worst = profit;
           }
        }
      if(index_worst == -1 || profit_worst >= 0)
         break;
      ulong ticket = PositionGetTicket(index_worst);
      trade.PositionClose(ticket);
      equity = AccountInfoDouble(ACCOUNT_EQUITY);
      drawdown = (max_equity - equity) / max_equity * 100.0;
     }
  }

//+------------------------------------------------------------------+
//| Calculates the number of bars and months in the tested period    |
//+------------------------------------------------------------------+
void CalculateBarsAndMonths()
  {
   datetime first = (datetime)SeriesInfoInteger(_Symbol,_Period,SERIES_FIRSTDATE);
   datetime last  = (datetime)SeriesInfoInteger(_Symbol,_Period,SERIES_LASTDATE);
   total_bars   = Bars(_Symbol,_Period,first,last);

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
   // nothing special
  }

//+------------------------------------------------------------------+

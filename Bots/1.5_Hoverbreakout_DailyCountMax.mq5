#property copyright "MJ Kruger"
#property link      "https://github.com/Maarten-Kruger/Trading"
#property version   "1.5"
#property strict

#include <Trade/Trade.mqh>

//--- input parameters
input int    InpRangeBars      = 10;     // Number of bars to define range (excluding breakout bar)
input double InpRangePoints    = 200;    // Maximum range size in points
input double InpTPPoints       = 400;    // Take profit distance in points
input double InpSLPoints       = 200;    // Stop loss distance in points
input double InpRiskPercent    = 1.0;    // Risk percentage of equity per trade
input int    InpMaxBarsOpen    = 5;      // Maximum bars to keep position open
input double InpMinDailyProfit = 1.0;    // Minimum daily profit to count as profitable day
uint   InpSlippage       = 5;      // Slippage in points
ulong  InpMagicNumber    = 1300001;// Magic number for HoverBreakout trades

//--- global objects
CTrade  trade;               // trading object

//--- test tracking variables
datetime g_test_start = 0;   // first bar time in test
datetime g_test_end   = 0;   // last bar time in test
int      g_total_bars = 0;   // number of processed bars

//+------------------------------------------------------------------+
//| Update test time and bar counters                                |
//+------------------------------------------------------------------+
void UpdateTestStats()
  {
   datetime now = TimeCurrent();
   if(g_test_start == 0)
      g_test_start = now;
   g_test_end = now;
   g_total_bars++;
  }

//+------------------------------------------------------------------+
//| Calculate number of months between two datetimes                 |
//+------------------------------------------------------------------+
double CalcMonths(datetime start_time, datetime end_time)
  {
   if(start_time == 0 || end_time == 0)
      return(0.0);

   MqlDateTime start_struct, end_struct;
   TimeToStruct(start_time, start_struct);
   TimeToStruct(end_time,   end_struct);

   int months = (end_struct.year - start_struct.year) * 12 +
                (end_struct.mon  - start_struct.mon)  + 1;
   return((double)months);
  }

//+------------------------------------------------------------------+
//| Helper: detect new bar                                          |
//+------------------------------------------------------------------+
bool IsNewBar()
  {
   static datetime last_bar_time = 0;
   datetime current_bar_time = iTime(_Symbol, _Period, 0);
   if(current_bar_time != last_bar_time)
     {
      last_bar_time = current_bar_time;
      return(true);
     }
   return(false);
  }

//+------------------------------------------------------------------+
//| Calculate hover range from past bars                            |
//| Returns true if bars stayed within specified range               |
//+------------------------------------------------------------------+
bool CalcRange(int bars_back, double &range_high, double &range_low)
  {
   if(bars_back < 2)
      return(false);

// initialise with second previous bar because bar1 is the breakout bar
   range_high = iHigh(_Symbol, _Period, 2);
   range_low  = iLow(_Symbol,  _Period, 2);

   for(int i = 3; i <= bars_back + 1; i++)
     {
      double h = iHigh(_Symbol, _Period, i);
      double l = iLow(_Symbol,  _Period, i);
      if(h > range_high)
         range_high = h;
      if(l < range_low)
         range_low  = l;
     }
   return((range_high - range_low) <= InpRangePoints * _Point);
  }

//+------------------------------------------------------------------+
//| Determine if Friday 16:00 GMT+2 close time has been reached      |
//+------------------------------------------------------------------+
bool IsFridayCutoffReached()
  {
   const int   GMT_PLUS_TWO = 2 * 60 * 60;
   datetime    now_gmt      = TimeGMT();
   datetime    now_gmt2     = now_gmt + GMT_PLUS_TWO;
   MqlDateTime time_struct;
   TimeToStruct(now_gmt2, time_struct);

   if(time_struct.day_of_week != 5) // 0 = Sunday ... 5 = Friday
      return(false);

   if(time_struct.hour > 16)
      return(true);

   if(time_struct.hour == 16 && time_struct.min >= 0)
      return(true);

   return(false);
  }

//+------------------------------------------------------------------+
//| Calculate trade volume based on equity risk                     |
//+------------------------------------------------------------------+
double CalcLotSize(double risk_percent)
  {
   double equity   = AccountInfoDouble(ACCOUNT_EQUITY);
   double tick_val = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size= SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double step     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double min_vol  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_vol  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

// prevent division by zero
   if(tick_val <= 0 || tick_size <= 0 || InpSLPoints <= 0)
      return(min_vol);

   double risk_amount   = equity * risk_percent / 100.0;
   double stop_distance = InpSLPoints * _Point;
   double ticks         = stop_distance / tick_size;
   double lot           = risk_amount / (ticks * tick_val);

// adjust to broker limits
   lot = MathFloor(lot / step) * step;
   lot = MathMax(min_vol, MathMin(max_vol, lot));

   return(lot);
  }

//+------------------------------------------------------------------+
//| Close open position after a number of bars                       |
//+------------------------------------------------------------------+
void CheckForExit()
  {
// Iterate through all open positions and close those that exceed the
// maximum number of bars specified in InpMaxBarsOpen.
   bool friday_cutoff = IsFridayCutoffReached();

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket))
         continue;

      string symbol = PositionGetString(POSITION_SYMBOL);
      if(symbol != _Symbol)
         continue;

      long magic = PositionGetInteger(POSITION_MAGIC);
      if(magic != (long)InpMagicNumber)
         continue;

      datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
      int bars_open = iBarShift(_Symbol, _Period, open_time);
      bool close_due_to_bars = (bars_open >= InpMaxBarsOpen && bars_open >= 0);

      if(friday_cutoff || close_due_to_bars)
         trade.PositionClose(ticket);
     }
  }

//+------------------------------------------------------------------+
//| Entry logic: trade on breakout from tight range                 |
//+------------------------------------------------------------------+
void CheckForEntry()
  {
   double high, low;
   if(!CalcRange(InpRangeBars, high, low))
      return; // Range condition not met

   if(IsFridayCutoffReached())
      return; // Do not open new trades after the cutoff time

   double last_close = iClose(_Symbol, _Period, 1);

   trade.SetDeviationInPoints(InpSlippage);

   double lots = CalcLotSize(InpRiskPercent);

// Breakout above the range -> buy
   if(last_close > high)
     {
      double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = price - InpSLPoints * _Point;
      double tp = price + InpTPPoints * _Point;
      trade.Buy(lots, _Symbol, price, sl, tp, "HoverBreakout");
     }
// Breakout below the range -> sell
   else
      if(last_close < low)
        {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl = price + InpSLPoints * _Point;
         double tp = price - InpTPPoints * _Point;
         trade.Sell(lots, _Symbol, price, sl, tp, "HoverBreakout");
        }
  }

//+------------------------------------------------------------------+
//| Expert initialization                                           |
//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber((long)InpMagicNumber);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   g_test_end = TimeCurrent();
  }

//+------------------------------------------------------------------+
//| Expert tick function                                            |
//+------------------------------------------------------------------+
void OnTick()
  {
// Only run logic once per new bar, loops otherwise
   bool new_bar = IsNewBar();
   if(!new_bar)
      return;

   UpdateTestStats();

   CheckForExit();   // manage existing position
   CheckForEntry();  // look for new opportunity
  }

//+------------------------------------------------------------------+
//| Custom optimization criterion                                    |
//+------------------------------------------------------------------+
double OnTester()
{
   HistorySelect(0, TimeCurrent());
   int total_deals = HistoryDealsTotal();

   double profitable_days_count = 0.0;

   if (total_deals == 0) return 0.0;

   // Identify the first day
   long first_ticket = HistoryDealGetTicket(0);
   datetime first_time = (datetime)HistoryDealGetInteger(first_ticket, DEAL_TIME);
   MqlDateTime current_date;
   TimeToStruct(first_time, current_date);

   int current_day_of_year = current_date.day_of_year;
   int current_year = current_date.year;

   double current_day_profit = 0.0;

   for(int i = 0; i < total_deals; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);

      long type = HistoryDealGetInteger(ticket, DEAL_TYPE);
      if (type == DEAL_TYPE_BALANCE) continue; // Skip deposits/withdrawals

      datetime time = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
      MqlDateTime dt;
      TimeToStruct(time, dt);

      bool is_same_day = (dt.day_of_year == current_day_of_year && dt.year == current_year);

      if (!is_same_day)
      {
         // Day changed. Process previous day.
         if (current_day_profit > InpMinDailyProfit)
         {
            profitable_days_count += 1.0;
         }

         // Reset for new day
         current_day_profit = 0.0;
         current_day_of_year = dt.day_of_year;
         current_year = dt.year;
      }

      double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
      double swap = HistoryDealGetDouble(ticket, DEAL_SWAP);
      double comm = HistoryDealGetDouble(ticket, DEAL_COMMISSION);

      current_day_profit += (profit + swap + comm);
   }

   // Process the last day
   if (current_day_profit > InpMinDailyProfit)
   {
      profitable_days_count += 1.0;
   }

   return profitable_days_count;
}

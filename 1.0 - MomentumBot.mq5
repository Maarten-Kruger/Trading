#property copyright ""
#property link      ""
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//--- input parameters for trade management
input double   InpRiskPercent      = 1.0;     // Risk per trade (% of equity)
input double   InpTakeProfitPips   = 60.0;    // Take-profit distance (pips)
input double   InpStopLossPips     = 40.0;    // Stop-loss distance (pips)
input int      InpMaxTimeMinutes   = 240;     // Maximum trade life (minutes)
input ulong    InpMagic            = 1200321; // Expert advisor magic number

//--- optimisation weighting inputs (must sum to 100%, but normalised internally)
input double   InpWeightTrades     = 40.0;    // Weight for trade density (Wt)
input double   InpWeightProfit     = 40.0;    // Weight for monthly profit (Wp)
input double   InpWeightDrawdown   = 20.0;    // Weight for drawdown (Wd)

//--- trading object
CTrade         g_trade;

//--- runtime state variables
ENUM_ORDER_TYPE g_next_direction   = ORDER_TYPE_BUY; // direction to use for the next trade
ENUM_ORDER_TYPE g_last_direction   = ORDER_TYPE_BUY; // direction used for the most recent trade
bool            g_should_open_trade = true;          // flag instructing the EA to place the next order
bool            g_use_random_direction = true;       // flag forcing the next trade to be random

datetime        g_test_start_time  = 0;  // first tick timestamp observed in the test
datetime        g_test_end_time    = 0;  // latest tick timestamp observed in the test

double          g_start_equity     = 0.0; // equity captured at the beginning of the test
int             g_total_trades     = 0;   // number of executed entries during the test
double          g_total_profit     = 0.0; // cumulative net profit realised during the test

//+------------------------------------------------------------------+
//| Helper: calculate pip to point conversion factor                 |
//+------------------------------------------------------------------+
double PipToPointFactor()
{
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   if(digits == 3 || digits == 5)
      return 10.0;
   return 1.0;
}

//+------------------------------------------------------------------+
//| Helper: choose a random order direction                          |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE RandomDirection()
{
   if(MathRand() % 2 == 0)
      return ORDER_TYPE_BUY;
   return ORDER_TYPE_SELL;
}

//+------------------------------------------------------------------+
//| Helper: get the opposite order direction                         |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE OppositeDirection(const ENUM_ORDER_TYPE direction)
{
   return (direction == ORDER_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
}

//+------------------------------------------------------------------+
//| Helper: check if EA currently has any open positions             |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;

      if(!PositionSelectByTicket(ticket))
         continue;

      if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagic)
         continue;

      if((string)PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;

      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Helper: close positions exceeding the maximum lifetime           |
//+------------------------------------------------------------------+
void EnforceMaxLife()
{
   if(InpMaxTimeMinutes <= 0)
      return;

   bool closed_by_time = false;
   datetime now = TimeCurrent();

   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;

      if(!PositionSelectByTicket(ticket))
         continue;

      if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagic)
         continue;

      if((string)PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;

      datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
      if(open_time == 0)
         continue;

      int lifetime_minutes = (int)((now - open_time) / 60);
      if(lifetime_minutes >= InpMaxTimeMinutes)
      {
         g_trade.PositionClose(ticket);
         closed_by_time = true;
      }
   }

   if(closed_by_time)
   {
      // restart the momentum train with a random direction when a time based exit occurs
      g_should_open_trade = true;
      g_use_random_direction = true;
   }
}

//+------------------------------------------------------------------+
//| Helper: calculate appropriate lot size from risk% and stop loss  |
//+------------------------------------------------------------------+
double CalculateLotSize()
{
   double stop_pips = MathMax(InpStopLossPips, 0.1);
   double point_factor = PipToPointFactor();
   double stop_distance_points = stop_pips * point_factor;
   double stop_distance_price = stop_distance_points * _Point;

   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

   if(tick_value <= 0.0 || tick_size <= 0.0 || stop_distance_price <= 0.0)
      return 0.0;

   double risk_amount = AccountInfoDouble(ACCOUNT_EQUITY) * InpRiskPercent / 100.0;
   double stop_value  = (stop_distance_price / tick_size) * tick_value;

   if(stop_value <= 0.0)
      return 0.0;

   double volume = risk_amount / stop_value;

   double min_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(step_volume <= 0.0)
      step_volume = min_volume;

   if(step_volume <= 0.0)
      return 0.0;

   volume = MathMax(volume, min_volume);
   volume = MathMin(volume, max_volume);
   volume = MathFloor(volume / step_volume) * step_volume;
   volume = MathMax(volume, min_volume);

   return NormalizeDouble(volume, (int)SymbolInfoInteger(_Symbol, SYMBOL_VOLUME_DIGITS));
}

//+------------------------------------------------------------------+
//| Helper: open a trade in the requested direction                   |
//+------------------------------------------------------------------+
bool OpenMomentumTrade(const ENUM_ORDER_TYPE direction)
{
   double volume = CalculateLotSize();
   if(volume <= 0.0)
      return false;

   double point_factor = PipToPointFactor();
   double sl_points = InpStopLossPips * point_factor;
   double tp_points = InpTakeProfitPips * point_factor;

   double price = 0.0, sl = 0.0, tp = 0.0;
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   if(direction == ORDER_TYPE_BUY)
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl = price - sl_points * _Point;
      tp = price + tp_points * _Point;
      if(!g_trade.Buy(volume, _Symbol, price, NormalizeDouble(sl, digits), NormalizeDouble(tp, digits)))
         return false;
   }
   else
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = price + sl_points * _Point;
      tp = price - tp_points * _Point;
      if(!g_trade.Sell(volume, _Symbol, price, NormalizeDouble(sl, digits), NormalizeDouble(tp, digits)))
         return false;
   }

   // remember the direction that has just been used for the next decision
   g_last_direction = direction;
   g_should_open_trade = false;
   g_use_random_direction = false;

   return true;
}

//+------------------------------------------------------------------+
//| Helper: update test-span timestamps                               |
//+------------------------------------------------------------------+
void UpdateTestSpan()
{
   datetime now = TimeCurrent();
   if(g_test_start_time == 0)
      g_test_start_time = now;
   g_test_end_time = now;
}

//+------------------------------------------------------------------+
//| Helper: compute span statistics for optimisation                 |
//+------------------------------------------------------------------+
void ComputeSpanStats(int &bars, double &months)
{
   bars = 1;
   months = 1.0 / 30.0; // default to ~one day to avoid division by zero

   if(g_test_start_time == 0 || g_test_end_time <= g_test_start_time)
      return;

   int calculated_bars = Bars(_Symbol, PERIOD_CURRENT, g_test_start_time, g_test_end_time);
   if(calculated_bars > 0)
      bars = calculated_bars;

   double seconds = (double)(g_test_end_time - g_test_start_time);
   if(seconds > 0.0)
      months = MathMax(seconds / (30.0 * 24.0 * 60.0 * 60.0), 1.0 / 30.0);
}

//+------------------------------------------------------------------+
//| Expert initialisation                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   MathSrand((uint)TimeLocal());

   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetTypeFilling(ORDER_FILLING_FOK);
   g_trade.SetTypeFillingBySymbol(_Symbol);

   g_start_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   g_total_trades = 0;
   g_total_profit = 0.0;

   g_test_start_time = 0;
   g_test_end_time = 0;

   g_should_open_trade = true;
   g_use_random_direction = true;
   g_next_direction = ORDER_TYPE_BUY;
   g_last_direction = ORDER_TYPE_BUY;

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialisation                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // No special clean-up required. Place-holder to satisfy structure guidelines.
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   UpdateTestSpan();
   EnforceMaxLife();

   if(HasOpenPosition())
      return;

   if(!g_should_open_trade)
      return;

   ENUM_ORDER_TYPE direction = g_use_random_direction ? RandomDirection() : g_next_direction;
   if(OpenMomentumTrade(direction))
      return;

   // If execution failed we will try again on the next tick.
}

//+------------------------------------------------------------------+
//| Trade transaction handler                                        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD)
      return;

   ulong deal_ticket = trans.deal;
   if(deal_ticket == 0)
      return;

   if(HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) != (long)InpMagic)
      return;

   ENUM_DEAL_ENTRY entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);
   if(entry == DEAL_ENTRY_IN)
   {
      ++g_total_trades;
      return;
   }

   if(entry != DEAL_ENTRY_OUT)
      return;

   double deal_profit = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT)
                      + HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION)
                      + HistoryDealGetDouble(deal_ticket, DEAL_SWAP);
   g_total_profit += deal_profit;

   // evaluate how to continue the momentum train
   if(deal_profit > 0.0)
   {
      g_next_direction = g_last_direction;
      g_should_open_trade = true;
      g_use_random_direction = false;
   }
   else if(deal_profit < 0.0)
   {
      g_next_direction = OppositeDirection(g_last_direction);
      g_should_open_trade = true;
      g_use_random_direction = false;
   }
   else
   {
      g_should_open_trade = true;
      g_use_random_direction = true;
   }
}

//+------------------------------------------------------------------+
//| Custom tester function                                           |
//+------------------------------------------------------------------+
double OnTester()
{
   int bars = 1;
   double months = 1.0 / 30.0;
   ComputeSpanStats(bars, months);

   double trade_density = (bars > 0) ? (double)g_total_trades / (double)bars : 0.0;
   double monthly_profit = 0.0;

   if(g_start_equity > 0.0 && months > 0.0)
      monthly_profit = (g_total_profit / g_start_equity) / months;

   double drawdown = TesterStatistics(STAT_EQUITY_DDREL_PERCENT) / 100.0;

   double weight_sum = InpWeightTrades + InpWeightProfit + InpWeightDrawdown;
   if(weight_sum <= 0.0)
      weight_sum = 100.0;

   double wt = InpWeightTrades / weight_sum;
   double wp = InpWeightProfit / weight_sum;
   double wd = InpWeightDrawdown / weight_sum;

   double score = trade_density * wt + monthly_profit * wp - drawdown * wd;

   return score * 100.0;
}

//+------------------------------------------------------------------+
git status -sb

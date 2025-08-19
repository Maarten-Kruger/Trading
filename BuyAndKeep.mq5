#property copyright "MJ Kruger"
#property link      "https://github.com/Maarten-Kruger/Trading"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//--- input parameters
input int    InpCandlesBack      = 20;      // Number of candles to average
input int    InpEntryDeviation   = 100;     // Entry threshold below average (points)
input int    InpTakeProfitFactor = 5;       // TP distance in multiples of average candle size
input double InpRiskPercent      = 1.0;     // Percent of equity to risk per trade
input double InpMaxDrawdown      = 30.0;    // Max equity drawdown percentage before closing all

CTrade trade;                 // trading object

//--- variables for drawdown control
 double g_peak_equity = 0.0;

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
//| Calculate lot size based on equity percentage                   |
//+------------------------------------------------------------------+
double CalculateLotSize()
  {
   double equity      = AccountInfoDouble(ACCOUNT_EQUITY);
   double lot_step    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double min_lot     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double volume      = equity * InpRiskPercent / 100.0 / 100000.0; // basic approximation
   volume             = MathMax(min_lot, MathMin(max_lot, MathFloor(volume/lot_step)*lot_step));
   return(volume);
  }

//+------------------------------------------------------------------+
//| Close all open positions                                        |
//+------------------------------------------------------------------+
void CloseAllPositions()
  {
   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      string symbol = PositionGetSymbol(i);
      if(SymbolInfoInteger(symbol, SYMBOL_SELECT))
        {
         trade.PositionClose(symbol);
        }
     }
  }

//+------------------------------------------------------------------+
//| Check account drawdown and close all if necessary               |
//+------------------------------------------------------------------+
void CheckDrawdown()
  {
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity > g_peak_equity)
      g_peak_equity = equity;
   double dd_percent = (g_peak_equity - equity) / g_peak_equity * 100.0;
   if(g_peak_equity > 0 && dd_percent >= InpMaxDrawdown)
     {
      CloseAllPositions();
      g_peak_equity = equity; // reset peak after closing
     }
  }

//+------------------------------------------------------------------+
//| Expert initialization                                           |
//+------------------------------------------------------------------+
int OnInit()
  {
   g_peak_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                            |
//+------------------------------------------------------------------+
void OnTick()
  {
   //--- risk control on every tick
   CheckDrawdown();

   if(!IsNewBar())
      return;

   //--- calculate average price and range of the past candles
   double avg_price = 0.0;
   double avg_range = 0.0;
   for(int i=1; i<=InpCandlesBack; i++)
     {
      avg_price += iClose(_Symbol, _Period, i);
      avg_range += (iHigh(_Symbol, _Period, i) - iLow(_Symbol, _Period, i));
     }
   avg_price /= InpCandlesBack;
   avg_range /= InpCandlesBack;

   //--- entry condition
   double open_price = iOpen(_Symbol, _Period, 0);
   if(open_price <= avg_price - InpEntryDeviation * _Point)
     {
      double tp_price = open_price + avg_range * InpTakeProfitFactor;
      double volume   = CalculateLotSize();

      trade.Buy(volume, _Symbol, open_price, 0.0, tp_price);
     }
  }

//+------------------------------------------------------------------+
//| Expert deinitialization                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }


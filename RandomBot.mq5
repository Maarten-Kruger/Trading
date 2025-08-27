#property copyright "MJ Kruger"
#property link      "https://github.com/Maarten-Kruger/Trading"
#property version   "2.00"
#property strict

#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>

//--- trading inputs
input int      InpBarsBetweenTrades = 10;    // Bars between random entries (X)
input int      InpStopLossPoints    = 200;   // Stop Loss in points (Z)
input int      InpTakeProfitPoints  = 200;   // Take Profit in points (Y)
input int      InpMaxBarsInTrade    = 20;    // Max bars to hold a trade
input double   InpRiskPercent       = 1.0;   // Risk per trade (% of equity)
input uint     InpSlippage          = 10;    // Slippage in points
input long     InpMagic             = 20250827; // Magic number

//--- optimization weights (should sum to 100)
input double   InpW_T               = 33.34; // Weight: Trade density (T)
input double   InpW_P               = 33.33; // Weight: Monthly profit (P)
input double   InpW_D               = 33.33; // Weight: Drawdown (D)

CTrade  trade;               // trading object
CPositionInfo pos_info;      // position info helper

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
//| Helper: bars since a given time                                  |
//+------------------------------------------------------------------+
int BarsSince(const datetime t)
  {
   int idx = iBarShift(_Symbol, _Period, t, false);
   if(idx < 0)
      return(0);
   return(idx);
  }

//+------------------------------------------------------------------+
//| Helper: month difference between two times                       |
//+------------------------------------------------------------------+
int MonthsDiff(const datetime from_time, const datetime to_time)
  {
   MqlDateTime a, b;
   TimeToStruct(from_time, a);
   TimeToStruct(to_time, b);
   int months = (b.year - a.year) * 12 + (b.mon - a.mon);
   if(b.day < a.day) months -= 1; // align partial month downwards
   if(months < 1) months = 1;     // avoid zero in optimization
   return(months);
  }

//+------------------------------------------------------------------+
//| Helper: get bars and months for tested period                    |
//+------------------------------------------------------------------+
void GetBarsAndMonths(int &total_bars, int &months)
  {
   total_bars = Bars(_Symbol, _Period);
   if(total_bars <= 0)
     {
      total_bars = 0;
      months = 1;
      return;
     }
   datetime first_bar = iTime(_Symbol, _Period, total_bars - 1);
   datetime last_bar  = iTime(_Symbol, _Period, 0);
   months = MonthsDiff(first_bar, last_bar);
  }

//+------------------------------------------------------------------+
//| Helper: volume by risk (% equity) and SL distance                |
//+------------------------------------------------------------------+
double CalcVolumeByRisk(const double risk_percent, const int sl_points)
  {
   if(risk_percent <= 0.0 || sl_points <= 0)
      return(0.0);

   double equity     = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_money = equity * (risk_percent / 100.0);

   double tick_value = 0.0, tick_size = 0.0;
   if(!SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE, tick_value)) return(0.0);
   if(!SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE,  tick_size )) return(0.0);

   // Money move per 1.0 price unit per 1 lot
   double money_per_price_unit = (tick_size > 0.0 ? (tick_value / tick_size) : 0.0);
   if(money_per_price_unit <= 0.0) return(0.0);

   double sl_price_distance = sl_points * _Point; // price units
   double loss_per_lot_at_sl = sl_price_distance * money_per_price_unit; // money per lot
   if(loss_per_lot_at_sl <= 0.0) return(0.0);

   double vol = risk_money / loss_per_lot_at_sl;

   // Normalize to symbol volume constraints
   double min_lot = 0.0, max_lot = 0.0, lot_step = 0.0;
   if(!SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN,  min_lot))  return(0.0);
   if(!SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX,  max_lot))  return(0.0);
   if(!SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP, lot_step)) return(0.0);

   if(vol < min_lot) vol = min_lot;
   if(vol > max_lot) vol = max_lot;
   if(lot_step > 0.0)
      vol = MathFloor(vol / lot_step) * lot_step; // round down to step

   // Derive volume digits from lot step (portable across builds)
   int vol_digits = 2;
   if(lot_step > 0.0)
     {
      double d = -MathLog10(lot_step);
      vol_digits = (int)MathRound(d);
      if(vol_digits < 0) vol_digits = 0;
      if(vol_digits > 8) vol_digits = 8;
     }
   vol = NormalizeDouble(vol, vol_digits);
   if(vol < min_lot) return(0.0);
   return(vol);
  }

//+------------------------------------------------------------------+
//| Helper: place random trade with SL/TP                            |
//+------------------------------------------------------------------+
bool PlaceRandomTrade()
  {
   int direction = (MathRand() % 2 == 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   double volume = CalcVolumeByRisk(InpRiskPercent, InpStopLossPoints);
   if(volume <= 0.0)
      return(false);

   trade.SetDeviationInPoints(InpSlippage);
   trade.SetExpertMagicNumber(InpMagic);

   double ask = 0.0, bid = 0.0;
   if(!SymbolInfoDouble(_Symbol, SYMBOL_ASK, ask)) return(false);
   if(!SymbolInfoDouble(_Symbol, SYMBOL_BID, bid)) return(false);
   if(ask <= 0.0 || bid <= 0.0)
      return(false);

   double sl = 0.0, tp = 0.0;
   if(direction == ORDER_TYPE_BUY)
     {
      sl = ask - InpStopLossPoints * _Point;
      tp = ask + InpTakeProfitPoints * _Point;
      return(trade.Buy(volume, _Symbol, 0.0, sl, tp, "RandomBot BUY"));
     }
   else
     {
      sl = bid + InpStopLossPoints * _Point;
      tp = bid - InpTakeProfitPoints * _Point;
      return(trade.Sell(volume, _Symbol, 0.0, sl, tp, "RandomBot SELL"));
     }
  }

//+------------------------------------------------------------------+
//| Close positions that exceed max bars in trade                    |
//+------------------------------------------------------------------+
void EnforceTimeExit()
  {
   for(int i = PositionsTotal() - 1; i >= 0; --i)
     {
      if(!pos_info.SelectByIndex(i))
         continue;

      string sym   = pos_info.Symbol();
      long   magic = (long)pos_info.Magic();
      if(sym != _Symbol || magic != InpMagic)
         continue;

      datetime opentime = (datetime)pos_info.Time();
      int bars_held = BarsSince(opentime);
      if(bars_held >= InpMaxBarsInTrade)
        {
         ulong ticket = pos_info.Ticket();
         if(!trade.PositionClose((long)ticket))
           {
            // Fallback for netting accounts
            trade.PositionClose(sym);
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Expert initialization                                           |
//+------------------------------------------------------------------+
int OnInit()
  {
   MathSrand((uint)TimeLocal());
   trade.SetExpertMagicNumber(InpMagic);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }

//+------------------------------------------------------------------+
//| Expert tick function                                            |
//+------------------------------------------------------------------+
void OnTick()
  {
   // On every tick, enforce time-based exits on our positions
   EnforceTimeExit();

   // Only evaluate entries on new bars
   static int bars_since_entry = 0;
   if(!IsNewBar())
      return;

   // Increment bar counter and check if it's time to place a trade
   bars_since_entry++;
   if(bars_since_entry < InpBarsBetweenTrades)
      return;

   // Try to place a random trade with SL/TP and risk-based volume
   if(PlaceRandomTrade())
     {
      bars_since_entry = 0; // reset after a successful entry
     }
  }

//+------------------------------------------------------------------+
//| Custom optimization criterion                                    |
//| y = T*Wt + P*Wp - D*Wd, multiplied by 100                        |
//+------------------------------------------------------------------+
double OnTester()
  {
   int total_bars = 0, months = 1;
   GetBarsAndMonths(total_bars, months);

   double trades = TesterStatistics(STAT_TRADES);
   double T = (total_bars > 0 ? trades / (double)total_bars : 0.0);

   double total_profit = TesterStatistics(STAT_PROFIT);
   double starting_equity = TesterStatistics(STAT_INITIAL_DEPOSIT);
   double P = (starting_equity > 0.0 && months > 0 ? (total_profit / starting_equity) / (double)months : 0.0);

   double D = TesterStatistics(STAT_EQUITY_DDREL_PERCENT) / 100.0; // convert percent to fraction

   double wt = InpW_T / 100.0;
   double wp = InpW_P / 100.0;
   double wd = InpW_D / 100.0;

   double y = (T * wt + P * wp - D * wd) * 100.0; // scale for readability
   return(y);
  }

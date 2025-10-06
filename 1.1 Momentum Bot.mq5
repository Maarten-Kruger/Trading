#property copyright ""
#property link      ""
#property version   "1.10"
#property strict

#include <Trade/Trade.mqh>

input ulong   InpMagicNumber            = 11001100;   // Expert magic number
input int     InpDSIPeriod              = 20;         // DSI lookback period
input double  InpDSIThreshold           = 55.0;       // Absolute DSI threshold
input int     InpLRPeriod               = 34;         // Linear regression lookback
input double  InpLRSlopeThreshold       = 0.00010;    // Absolute LR slope threshold
input double  InpRiskPerTrade           = 1.0;        // Risk per trade (% equity)
input double  InpStopLossPoints         = 800.0;      // Stop loss distance (points)
input double  InpTakeProfitPoints       = 1200.0;     // Take profit distance (points)
input int     InpMaxBarsHolding         = 60;         // Maximum holding time (bars, 0 = disabled)
input int     InpFridayCloseHour        = 16;         // Friday closing hour (GMT+offset)
input int     InpFridayCloseMinute      = 0;          // Friday closing minute (GMT+offset)
input int     InpGMTCloseOffsetHours    = 2;          // GMT offset used for Friday closing check
input double  InpWeightTradeDensity     = 34.0;       // Weight - trade density (%)
input double  InpWeightMonthlyProfit    = 33.0;       // Weight - monthly return (%)
input double  InpWeightDrawdown         = 33.0;       // Weight - drawdown (%)

CTrade        g_trade;                   // Trading helper

datetime      g_currentBarOpen = 0;      // Tracks the latest known bar open time
datetime      g_firstCompletedBar = 0;   // Timestamp of the first completed bar

datetime      g_lastCompletedBar = 0;    // Timestamp of the last completed bar
ulong         g_completedBarCount = 0;    // Number of completed bars observed

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
  {
   g_trade.SetExpertMagicNumber((long)InpMagicNumber);
   g_currentBarOpen    = 0;
   g_firstCompletedBar = 0;
   g_lastCompletedBar  = 0;
   g_completedBarCount = 0;
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Manage existing positions each tick before looking for new trades
   ManageFridayClose();
   ManageTimeStops();

   if(IsNewBar())
     {
      EvaluateSignals();
     }
  }

//+------------------------------------------------------------------+
//| Determines if a new bar has formed                               |
//+------------------------------------------------------------------+
bool IsNewBar()
  {
   datetime currentOpen = iTime(_Symbol,_Period,0);
   if(currentOpen<=0)
      return(false);

   if(g_currentBarOpen==0)
     {
      g_currentBarOpen = currentOpen;
      return(false);
     }

   if(currentOpen!=g_currentBarOpen)
     {
      // A new bar is now active, update statistics for the completed bar
      datetime completedBarTime = iTime(_Symbol,_Period,1);
      if(completedBarTime>0)
        {
         g_lastCompletedBar = completedBarTime;
         if(g_completedBarCount==0)
            g_firstCompletedBar = completedBarTime;
         g_completedBarCount++;
        }

      g_currentBarOpen = currentOpen;
      return(true);
     }

   return(false);
  }

//+------------------------------------------------------------------+
//| Main signal evaluation routine                                   |
//+------------------------------------------------------------------+
void EvaluateSignals()
  {
   // Acquire the closed bar's momentum readings
   double dsi   = CalculateDSI(InpDSIPeriod,1);
   double slope = CalculateLinearRegressionSlope(InpLRPeriod,1);

   if(!DoubleIsValid(dsi) || !DoubleIsValid(slope))
      return;

   double absThreshold = MathAbs(InpDSIThreshold);
   double absSlopeTh   = MathAbs(InpLRSlopeThreshold);

   // Long setup: positive DSI and positive slope above thresholds
   if(dsi>=absThreshold && slope>=absSlopeTh)
      EnterPosition(ORDER_TYPE_BUY);

   // Short setup: negative DSI and negative slope beyond thresholds
   if(dsi<=-absThreshold && slope<=-absSlopeTh)
      EnterPosition(ORDER_TYPE_SELL);
  }

//+------------------------------------------------------------------+
//| Places a new trade in the indicated direction                    |
//+------------------------------------------------------------------+
void EnterPosition(const ENUM_ORDER_TYPE orderType)
  {
   double stopLossPoints = MathMax(InpStopLossPoints,0.0);
   double takeProfitPoints = MathMax(InpTakeProfitPoints,0.0);

   double volume = CalculatePositionVolume(stopLossPoints);
   if(volume<=0.0)
     {
      Print(__FUNCTION__,": volume calculation failed");
      return;
     }

   double price = (orderType==ORDER_TYPE_BUY ? SymbolInfoDouble(_Symbol,SYMBOL_ASK) : SymbolInfoDouble(_Symbol,SYMBOL_BID));
   if(price<=0.0)
      return;

   double point      = _Point;
   double slDistance = stopLossPoints * point;
   double tpDistance = takeProfitPoints * point;

   double sl = 0.0;
   double tp = 0.0;

   if(stopLossPoints>0.0)
     {
      sl = (orderType==ORDER_TYPE_BUY ? price - slDistance : price + slDistance);
      sl = NormalizeDouble(sl,_Digits);
     }

   if(takeProfitPoints>0.0)
     {
      tp = (orderType==ORDER_TYPE_BUY ? price + tpDistance : price - tpDistance);
      tp = NormalizeDouble(tp,_Digits);
     }

   string comment = "MomentumBot1.1";
   bool sent = false;

   if(orderType==ORDER_TYPE_BUY)
      sent = g_trade.Buy(volume,_Symbol,price,sl,tp,comment);
   else if(orderType==ORDER_TYPE_SELL)
      sent = g_trade.Sell(volume,_Symbol,price,sl,tp,comment);

   if(!sent)
      Print(__FUNCTION__,": order send failed. Error = ",GetLastError());
  }

//+------------------------------------------------------------------+
//| Calculates the DSI for the specified shift                        |
//+------------------------------------------------------------------+
double CalculateDSI(const int period,const int shift)
  {
   if(period<=0)
      return(0.0);

   double closeBuffer[];
   ArraySetAsSeries(closeBuffer,true);

   if(CopyClose(_Symbol,_Period,shift,period+1,closeBuffer)!=(period+1))
      return(EMPTY_VALUE);

   double sumDiff = 0.0;
   double sumAbs  = 0.0;

   for(int i=0;i<period;i++)
     {
      double diff = closeBuffer[i] - closeBuffer[i+1];
      sumDiff += diff;
      sumAbs  += MathAbs(diff);
     }

   if(sumAbs==0.0)
      return(0.0);

   double dsi = 100.0 * (sumDiff / sumAbs);
   return(dsi);
  }

//+------------------------------------------------------------------+
//| Calculates the slope of a linear regression                       |
//+------------------------------------------------------------------+
double CalculateLinearRegressionSlope(const int period,const int shift)
  {
   if(period<2)
      return(0.0);

   double priceBuffer[];
   ArraySetAsSeries(priceBuffer,true);

   if(CopyClose(_Symbol,_Period,shift,period,priceBuffer)!=period)
      return(EMPTY_VALUE);

   double sumX = 0.0;
   double sumY = 0.0;
   double sumXY = 0.0;
   double sumXX = 0.0;

   for(int i=0;i<period;i++)
     {
      double x = (double)(period - 1 - i);
      double y = priceBuffer[i];
      sumX  += x;
      sumY  += y;
      sumXY += x*y;
      sumXX += x*x;
     }

   double n     = (double)period;
   double denom = (n*sumXX - sumX*sumX);
   if(denom==0.0)
      return(0.0);

   double slope = (n*sumXY - sumX*sumY) / denom;
   return(slope);
  }

//+------------------------------------------------------------------+
//| Computes a trade volume based on risk and stop-loss distance      |
//+------------------------------------------------------------------+
double CalculatePositionVolume(const double stopLossPoints)
  {
   double minVolume   = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   double maxVolume   = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX);
   double stepVolume  = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   double tickValue   = SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE);
   double tickSize    = SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);

   if(minVolume<=0.0 || maxVolume<=0.0 || tickValue<=0.0 || tickSize<=0.0)
      return(0.0);

   double equity      = AccountInfoDouble(ACCOUNT_EQUITY);
   double riskMoney   = equity * InpRiskPerTrade / 100.0;

   if(riskMoney<=0.0)
      return(0.0);

   double stopInPrice = stopLossPoints * _Point;
   if(stopInPrice<=0.0)
     {
      // If no stop distance is defined, fall back to minimum volume
      return(NormalizeVolume(minVolume,minVolume,stepVolume,maxVolume));
     }

   double valuePerLot = (tickValue / tickSize) * stopInPrice;
   if(valuePerLot<=0.0)
      return(0.0);

   double rawVolume = riskMoney / valuePerLot;
   return(NormalizeVolume(rawVolume,minVolume,stepVolume,maxVolume));
  }

//+------------------------------------------------------------------+
//| Normalizes trade volume respecting broker constraints             |
//+------------------------------------------------------------------+
double NormalizeVolume(const double volume,const double minVolume,const double step,const double maxVolume)
  {
   double clipped = MathMax(minVolume,MathMin(volume,maxVolume));
   if(step<=0.0)
      return(NormalizeDouble(clipped,2));

   double stepsCount = MathFloor((clipped - minVolume)/step + 0.5);
   double normalized = minVolume + stepsCount*step;
   normalized = MathMax(minVolume,MathMin(normalized,maxVolume));

   int volumeDigits = 2;
   if(step>0.0)
     {
      double stepDigits = -MathLog10(step);
      if(MathIsValidNumber(stepDigits))
        {
         volumeDigits = (int)MathRound(stepDigits);
         volumeDigits = (int)MathMax(0.0,MathMin((double)volumeDigits,8.0));
        }
     }
   return(NormalizeDouble(normalized,volumeDigits));
  }

//+------------------------------------------------------------------+
//| Closes trades that exceed their maximum holding period            |
//+------------------------------------------------------------------+
void ManageTimeStops()
  {
   if(InpMaxBarsHolding<=0)
      return;

   double periodSeconds = PeriodSeconds(_Period);
   if(periodSeconds<=0.0)
      return;

   for(int i=PositionsTotal()-1;i>=0;--i)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket==0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber)
         continue;

      datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);
      double heldBars = (TimeCurrent() - openTime) / periodSeconds;
      if(heldBars>=InpMaxBarsHolding)
        {
         if(!g_trade.PositionClose(ticket))
            Print(__FUNCTION__,": failed to close position ",ticket," on time-stop. Error = ",GetLastError());
        }
     }
  }

//+------------------------------------------------------------------+
//| Closes all positions on Friday at the configured time             |
//+------------------------------------------------------------------+
void ManageFridayClose()
  {
   if(!IsFridayCloseWindow())
      return;

   for(int i=PositionsTotal()-1;i>=0;--i)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket==0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber)
         continue;

      if(!g_trade.PositionClose(ticket))
         Print(__FUNCTION__,": failed to close position ",ticket," on Friday exit. Error = ",GetLastError());
     }
  }

//+------------------------------------------------------------------+
//| Determines if the Friday closing rule is active                   |
//+------------------------------------------------------------------+
bool IsFridayCloseWindow()
  {
   datetime gmtTime = TimeGMT();
   gmtTime += (InpGMTCloseOffsetHours * 3600);

   MqlDateTime dt;
   TimeToStruct(gmtTime,dt);

   if(dt.day_of_week!=5) // 5 = Friday
      return(false);

   if(dt.hour>InpFridayCloseHour)
      return(true);
   if(dt.hour==InpFridayCloseHour && dt.min>=InpFridayCloseMinute)
      return(true);

   return(false);
  }

//+------------------------------------------------------------------+
//| Custom tester function - objective metric                         |
//+------------------------------------------------------------------+
double OnTester()
  {
   double bars = (double)g_completedBarCount;
   if(bars<=0.0)
      bars = 1.0;

   double months = CalculateElapsedMonths(g_firstCompletedBar,g_lastCompletedBar);
   if(months<=0.0)
      months = 1.0;

   double totalTrades   = TesterStatistics(STAT_TRADES);
   double totalProfit   = TesterStatistics(STAT_PROFIT);
   double initialEquity = TesterStatistics(STAT_INITIAL_DEPOSIT);
   double drawdownRel   = TesterStatistics(STAT_EQUITY_DDREL_PERCENT) / 100.0;

   double tradeDensity  = totalTrades / bars;
   double monthlyReturn = 0.0;
   if(initialEquity>0.0)
      monthlyReturn = (totalProfit / initialEquity) / months;

   double wT = InpWeightTradeDensity / 100.0;
   double wP = InpWeightMonthlyProfit / 100.0;
   double wD = InpWeightDrawdown / 100.0;

   double objective = tradeDensity * wT + monthlyReturn * wP - drawdownRel * wD;
   return(objective * 100.0);
  }

//+------------------------------------------------------------------+
//| Calculates the number of months between two timestamps            |
//+------------------------------------------------------------------+
double CalculateElapsedMonths(const datetime startTime,const datetime endTime)
  {
   if(startTime<=0 || endTime<=startTime)
      return(0.0);

   double seconds = (double)(endTime - startTime);
   if(seconds<=0.0)
      return(0.0);

   const double averageMonthSeconds = 30.4375 * 24.0 * 3600.0;
   return(seconds / averageMonthSeconds);
  }

//+------------------------------------------------------------------+
//| Utility to validate numerical calculations                        |
//+------------------------------------------------------------------+
bool DoubleIsValid(const double value)
  {
   if(!MathIsValidNumber(value))
      return(false);

   if(value==EMPTY_VALUE || value==DBL_MAX || value==-DBL_MAX)
      return(false);

   return(true);
  }

//+------------------------------------------------------------------+

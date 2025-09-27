#property copyright ""
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//+------------------------------------------------------------------+
//| Expert Advisor inputs                                            |
//+------------------------------------------------------------------+
input string InpSymbol               = _Symbol;   // Trading symbol
input double InpRiskPercent          = 1.0;       // Risk per trade (% of equity)
input double InpTakeProfitPips       = 50.0;      // Take-profit distance (pips)
input double InpStopLossPips         = 30.0;      // Stop-loss distance (pips)
input int    InpMaxTimeMinutes       = 240;       // Maximum trade duration (minutes)
input int    InpMomentumBars         = 3;         // Bars to evaluate for momentum reset
input ulong  InpMagicNumber          = 20241011;  // Magic number for the expert
input double InpWeightTrades         = 40.0;      // Weight of trade density (sum must be 100%)
input double InpWeightProfit         = 40.0;      // Weight of monthly profit (sum must be 100%)
input double InpWeightDrawdown       = 20.0;      // Weight of drawdown impact (sum must be 100%)

//+------------------------------------------------------------------+
//| Global state                                                     |
//+------------------------------------------------------------------+
CTrade   g_trade;                     // Trading helper object
long     g_activePositionTicket = -1; // Ticket of the currently managed position
ENUM_ORDER_TYPE g_currentDirection;   // Direction we aim to trade next
bool     g_waitingForExit = false;    // True if we are waiting for the last trade to finish
bool     g_timeoutTriggered = false;  // Tracks whether the last closure was due to timeout

datetime g_tradeOpenTime = 0;         // Time the managed position was opened

// Tracking variables for tester statistics
static datetime g_testStart = 0;
static datetime g_testEnd   = 0;
static ulong    g_totalBars = 0;
static datetime g_lastBarTime = 0;

//+------------------------------------------------------------------+
//| Forward declarations                                             |
//+------------------------------------------------------------------+
void        InitializeDirection();
void        EvaluateMomentumDirection();
void        TryOpenNewTrade();
bool        PositionIsActive();
long        FindActivePositionTicket();
void        MonitorOpenPosition();
void        HandlePositionClosure(long ticket);
bool        ClosePositionByTicket(long ticket);
double      PipToPointMultiplier();
double      NormalizeLotSize(double lots);
double      CalculatePositionSize(ENUM_ORDER_TYPE orderType,double slPrice,double entryPrice);
void        UpdateBacktestTracking();
double      CalculateMonths(datetime start_time,datetime end_time);

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
  {
   MathSrand((uint)TimeLocal());
   if(!SymbolSelect(InpSymbol,true))
      return(INIT_FAILED);
   InitializeDirection();
   g_trade.SetExpertMagicNumber((ulong)InpMagicNumber);
   g_trade.SetAsyncMode(false);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Capture the last known end time when the expert stops
   g_testEnd=TimeCurrent();
  }

//+------------------------------------------------------------------+
//| Expert tick handler                                              |
//+------------------------------------------------------------------+
void OnTick()
  {
   UpdateBacktestTracking();

   if(!PositionIsActive())
     {
      if(g_waitingForExit)
        {
         // The previously managed trade has closed, handle the outcome
         HandlePositionClosure(g_activePositionTicket);
         g_waitingForExit=false;
        }

      TryOpenNewTrade();
     }
   else
     {
      MonitorOpenPosition();
     }
  }

//+------------------------------------------------------------------+
//| Update backtest tracking for bars and time                       |
//+------------------------------------------------------------------+
void UpdateBacktestTracking()
  {
   datetime now=TimeCurrent();
   if(g_testStart==0)
      g_testStart=now;
   g_testEnd=now;

   datetime currentBarTime=iTime(InpSymbol,_Period,0);
   if(currentBarTime!=g_lastBarTime && currentBarTime>0)
     {
      g_totalBars++;
      g_lastBarTime=currentBarTime;
     }
  }

//+------------------------------------------------------------------+
//| Determine the initial trade direction randomly                   |
//+------------------------------------------------------------------+
void InitializeDirection()
  {
   if(MathRand()%2==0)
      g_currentDirection=ORDER_TYPE_BUY;
   else
      g_currentDirection=ORDER_TYPE_SELL;
  }

//+------------------------------------------------------------------+
//| Evaluate direction based on recent momentum                      |
//+------------------------------------------------------------------+
void EvaluateMomentumDirection()
  {
   int bullish=0;
   int bearish=0;

   for(int i=1;i<=InpMomentumBars && i<Bars(InpSymbol,_Period);i++)
     {
      double close_i=iClose(InpSymbol,_Period,i-1);
      double close_prev=iClose(InpSymbol,_Period,i);
      if(close_i>close_prev)
         bullish++;
      else if(close_i<close_prev)
         bearish++;
     }

   if(bullish>bearish)
      g_currentDirection=ORDER_TYPE_BUY;
   else if(bearish>bullish)
      g_currentDirection=ORDER_TYPE_SELL;
   else
      InitializeDirection();
  }

//+------------------------------------------------------------------+
//| Attempt to open a new trade in the current direction             |
//+------------------------------------------------------------------+
void TryOpenNewTrade()
  {
   double pointMultiplier=PipToPointMultiplier();
   double slPips=InpStopLossPips;
   double tpPips=InpTakeProfitPips;

   if(slPips<=0.0 || tpPips<=0.0 || pointMultiplier<=0.0)
      return;

   MqlTick tick;
   if(!SymbolInfoTick(InpSymbol,tick))
      return;

   double slPrice, tpPrice;
   double price;

   if(g_currentDirection==ORDER_TYPE_BUY)
     {
      price=tick.ask;
      slPrice=price-slPips*pointMultiplier;
      tpPrice=price+tpPips*pointMultiplier;
     }
   else
     {
      price=tick.bid;
      slPrice=price+slPips*pointMultiplier;
      tpPrice=price-tpPips*pointMultiplier;
     }

   int priceDigits=(int)SymbolInfoInteger(InpSymbol,SYMBOL_DIGITS);
   slPrice=NormalizeDouble(slPrice,priceDigits);
   tpPrice=NormalizeDouble(tpPrice,priceDigits);

   double lotSize=CalculatePositionSize(g_currentDirection,slPrice,price);
   if(lotSize<=0.0)
      return;

   bool result=false;
   if(g_currentDirection==ORDER_TYPE_BUY)
      result=g_trade.Buy(lotSize,InpSymbol,price,slPrice,tpPrice);
   else
      result=g_trade.Sell(lotSize,InpSymbol,price,slPrice,tpPrice);

   if(result)
     {
      g_activePositionTicket=g_trade.ResultDeal();
      if(g_activePositionTicket<=0)
         g_activePositionTicket=g_trade.ResultOrder();

      long positionTicket=FindActivePositionTicket();
      if(positionTicket>0)
         g_activePositionTicket=positionTicket;

      if(PositionSelectByTicket((ulong)g_activePositionTicket))
         g_tradeOpenTime=(datetime)PositionGetInteger(POSITION_TIME);
      else
         g_tradeOpenTime=TimeCurrent();
      g_waitingForExit=true;
      g_timeoutTriggered=false;
     }
  }

//+------------------------------------------------------------------+
//| Check if the managed position is still active                    |
//+------------------------------------------------------------------+
bool PositionIsActive()
  {
   if(g_activePositionTicket<=0)
      return(false);

   ulong total=PositionsTotal();
   for(ulong i=0;i<total;i++)
     {
      if(!PositionSelectByIndex((int)i))
         continue;

      long ticket=(long)PositionGetInteger(POSITION_TICKET);
      if(ticket==g_activePositionTicket)
         return(true);
     }

   return(false);
  }

//+------------------------------------------------------------------+
//| Locate the latest open position managed by this expert           |
//+------------------------------------------------------------------+
long FindActivePositionTicket()
  {
   ulong total=PositionsTotal();
   for(ulong i=0;i<total;i++)
     {
      if(!PositionSelectByIndex((int)i))
         continue;

      if((long)PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber)
         continue;

      string symbol="";
      PositionGetString(POSITION_SYMBOL,symbol);
      if(symbol!=InpSymbol)
         continue;

      return((long)PositionGetInteger(POSITION_TICKET));
     }

   return(-1);
  }

//+------------------------------------------------------------------+
//| Monitor the active position for timeout condition                |
//+------------------------------------------------------------------+
void MonitorOpenPosition()
  {
   if(g_tradeOpenTime==0)
      return;

   int maxSeconds=InpMaxTimeMinutes*60;
   if(maxSeconds<=0)
      return;

   datetime now=TimeCurrent();
   if((int)(now-g_tradeOpenTime)>=maxSeconds)
     {
      if(ClosePositionByTicket(g_activePositionTicket))
        {
         g_timeoutTriggered=true;
         g_waitingForExit=true;
        }
     }
  }

//+------------------------------------------------------------------+
//| Close position by ticket                                         |
//+------------------------------------------------------------------+
bool ClosePositionByTicket(long ticket)
  {
   if(ticket<=0)
      return(false);

   if(!PositionSelectByTicket((ulong)ticket))
      return(false);

   ENUM_POSITION_TYPE type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

   bool closed=false;
   if(type==POSITION_TYPE_BUY)
      closed=g_trade.PositionClose(ticket);
   else if(type==POSITION_TYPE_SELL)
      closed=g_trade.PositionClose(ticket);

   if(closed)
     {
      g_tradeOpenTime=0;
      return(true);
     }

   return(false);
  }

//+------------------------------------------------------------------+
//| Handle the closure of the managed position                       |
//+------------------------------------------------------------------+
void HandlePositionClosure(long ticket)
  {
   if(ticket<=0)
     {
      EvaluateMomentumDirection();
      return;
     }

   if(g_timeoutTriggered)
     {
      EvaluateMomentumDirection();
      g_activePositionTicket=-1;
      g_tradeOpenTime=0;
      g_timeoutTriggered=false;
      return;
     }

   datetime historyStart=g_tradeOpenTime;
   if(historyStart>0)
      historyStart-=PeriodSeconds(_Period)*2;
   if(historyStart<0)
      historyStart=0;

   if(!HistorySelect(historyStart,TimeCurrent()))
      HistorySelect(0,TimeCurrent());

   ulong deals=HistoryDealsTotal();
   double lastProfit=0.0;
   datetime lastCloseTime=0;

   for(ulong i=0;i<deals;i++)
     {
      ulong dealTicket=HistoryDealGetTicket(i);
      if((long)HistoryDealGetInteger(dealTicket,DEAL_MAGIC)!= (long)InpMagicNumber)
         continue;

      if(HistoryDealGetInteger(dealTicket,DEAL_ENTRY)==DEAL_ENTRY_OUT)
        {
         datetime closeTime=(datetime)HistoryDealGetInteger(dealTicket,DEAL_TIME);
         string dealSymbol="";
         HistoryDealGetString(dealTicket,DEAL_SYMBOL,dealSymbol);
         if(dealSymbol!=InpSymbol)
            continue;

         if(closeTime>lastCloseTime)
           {
            lastCloseTime=closeTime;
            lastProfit=HistoryDealGetDouble(dealTicket,DEAL_PROFIT)+HistoryDealGetDouble(dealTicket,DEAL_SWAP)+HistoryDealGetDouble(dealTicket,DEAL_COMMISSION);
           }
        }
     }

   if(lastCloseTime==0)
     {
      EvaluateMomentumDirection();
     }
   else
     {
      if(lastProfit>0.0)
        {
         // Continue trading in the same direction after a profitable exit
        }
      else if(lastProfit<0.0)
        {
         g_currentDirection=(g_currentDirection==ORDER_TYPE_BUY ? ORDER_TYPE_SELL : ORDER_TYPE_BUY);
        }
      else
        {
         EvaluateMomentumDirection();
        }
     }

   g_activePositionTicket=-1;
   g_tradeOpenTime=0;
   g_timeoutTriggered=false;
  }

//+------------------------------------------------------------------+
//| Convert pips to price distance                                   |
//+------------------------------------------------------------------+
double PipToPointMultiplier()
  {
   double point=SymbolInfoDouble(InpSymbol,SYMBOL_POINT);
   if(point<=0.0)
      return(0.0);
   int digits=(int)SymbolInfoInteger(InpSymbol,SYMBOL_DIGITS);
   if(digits==3 || digits==5)
      return(point*10.0);
   return(point);
  }

//+------------------------------------------------------------------+
//| Normalize lot size respecting symbol settings                    |
//+------------------------------------------------------------------+
double NormalizeLotSize(double lots)
  {
   double step=SymbolInfoDouble(InpSymbol,SYMBOL_VOLUME_STEP);
   double min=SymbolInfoDouble(InpSymbol,SYMBOL_VOLUME_MIN);
   double max=SymbolInfoDouble(InpSymbol,SYMBOL_VOLUME_MAX);

   if(step<=0.0)
      step=1.0;

   double normalized=MathFloor(lots/step)*step;
   normalized=MathMax(normalized,min);
   normalized=MathMin(normalized,max);
   return(NormalizeDouble(normalized,(int)SymbolInfoInteger(InpSymbol,SYMBOL_VOLUME_DIGITS)));
  }

//+------------------------------------------------------------------+
//| Calculate position size based on risk                             |
//+------------------------------------------------------------------+
double CalculatePositionSize(ENUM_ORDER_TYPE orderType,double slPrice,double entryPrice)
  {
   double equity=AccountInfoDouble(ACCOUNT_EQUITY);
   double riskMoney=equity*InpRiskPercent/100.0;
   if(riskMoney<=0.0)
      return(0.0);

   double point=SymbolInfoDouble(InpSymbol,SYMBOL_POINT);
   double tickValue=SymbolInfoDouble(InpSymbol,SYMBOL_TRADE_TICK_VALUE);
   double tickSize=SymbolInfoDouble(InpSymbol,SYMBOL_TRADE_TICK_SIZE);

   if(tickSize<=0.0 || point<=0.0)
      return(0.0);

   double stopDistance=MathAbs(entryPrice-slPrice);
   if(stopDistance<=0.0)
      return(0.0);

   double valuePerPoint=tickValue/tickSize;
   double stopPoints=stopDistance/point;
   double riskPerLot=stopPoints*valuePerPoint;
   if(riskPerLot<=0.0)
      return(0.0);

   double lots=riskMoney/riskPerLot;
   lots=NormalizeLotSize(lots);

   return(lots);
  }

//+------------------------------------------------------------------+
//| Calculate the number of months between two dates                 |
//+------------------------------------------------------------------+
double CalculateMonths(datetime start_time,datetime end_time)
  {
   if(start_time<=0 || end_time<=start_time)
      return(1.0);

   double seconds=(double)(end_time-start_time);
   double months=seconds/(30.4375*24.0*60.0*60.0);
   if(months<=0.0)
      months=1.0;
   return(months);
  }

//+------------------------------------------------------------------+
//| Custom tester optimization target                                |
//+------------------------------------------------------------------+
double OnTester()
  {
   double totalTrades=TesterStatistics(STAT_TRADES);
   double totalProfit=TesterStatistics(STAT_PROFIT);
   double initialDeposit=TesterStatistics(STAT_INITIAL_DEPOSIT);
   double drawdown=TesterStatistics(STAT_EQUITY_DDREL_PERCENT)/100.0;

   double bars=(double)g_totalBars;
   if(bars<=0.0)
      bars=1.0;
   double tradeDensity=totalTrades/bars;

   double months=CalculateMonths(g_testStart,g_testEnd);
   double monthlyProfit=0.0;
   if(initialDeposit>0.0 && months>0.0)
      monthlyProfit=(totalProfit/initialDeposit)/months;

   double weightSum=InpWeightTrades+InpWeightProfit+InpWeightDrawdown;
   if(weightSum<=0.0)
      weightSum=100.0;

   double wt=InpWeightTrades/weightSum;
   double wp=InpWeightProfit/weightSum;
   double wd=InpWeightDrawdown/weightSum;

   double objective=(tradeDensity*wt + monthlyProfit*wp - drawdown*wd)*100.0;
   return(objective);
  }

//+------------------------------------------------------------------+

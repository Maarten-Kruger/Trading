#property copyright ""
#property version   "2.00"
#property strict

#include <Trade/Trade.mqh>

//+------------------------------------------------------------------+
//| Expert Advisor inputs                                            |
//+------------------------------------------------------------------+
input string InpSymbol                 = "";        // Trading symbol (leave blank for current)
input double InpRiskPercent            = 1.0;       // Risk per trade (% of equity)
input double InpTakeProfitPips         = 50.0;      // Take-profit distance (pips)
input double InpStopLossPips           = 30.0;      // Stop-loss distance (pips)
input int    InpMaxTimeMinutes         = 240;       // Maximum trade duration (minutes, 0 = disabled)
input int    InpMomentumBars           = 3;         // Bars to evaluate for momentum reset
input int    InpMaxConcurrentPositions = 2;         // Maximum simultaneous positions managed by the EA
input ulong  InpMagicNumber            = 20241011;  // Magic number for the expert
input double InpWeightTrades           = 40.0;      // Weight of trade density (sum must be 100%)
input double InpWeightProfit           = 40.0;      // Weight of monthly profit (sum must be 100%)
input double InpWeightDrawdown         = 20.0;      // Weight of drawdown impact (sum must be 100%)

//+------------------------------------------------------------------+
//| Internal structures and enumerations                             |
//+------------------------------------------------------------------+
enum MomentumOutcome
  {
   MOMENTUM_NONE = 0,
   MOMENTUM_PROFIT,
   MOMENTUM_LOSS,
   MOMENTUM_TIMEOUT
  };

struct PositionContext
  {
   ulong            position_id;   // Ticket of the position
   ENUM_ORDER_TYPE  direction;     // Original direction used for the entry
  };

//+------------------------------------------------------------------+
//| Global state                                                     |
//+------------------------------------------------------------------+
CTrade          g_trade;                               // Trading helper
string          g_symbol            = "";              // Cached trading symbol
ENUM_ORDER_TYPE g_currentDirection  = ORDER_TYPE_BUY;  // Direction selected for the next entry
int             g_slotsToFill       = 0;               // Remaining entry slots we may occupy

// Storage for active position metadata (supports hedging accounts)
#define MAX_MANAGED_POSITIONS 128
PositionContext g_positionBook[MAX_MANAGED_POSITIONS];
int             g_positionCount = 0;

// Storage for tickets closed due to timeout so we can handle the outcome correctly
ulong           g_timeoutTickets[MAX_MANAGED_POSITIONS];
int             g_timeoutCount = 0;

// Tracking variables for tester statistics
static datetime g_testStart   = 0;
static datetime g_testEnd     = 0;
static ulong    g_totalBars   = 0;
static datetime g_lastBarTime = 0;

// Miscellaneous helpers
datetime        g_lastEntryTime = 0;                   // Time of the most recent successful entry

//+------------------------------------------------------------------+
//| Forward declarations                                             |
//+------------------------------------------------------------------+
bool   EnsureSymbol();
void   InitializeDirection();
void   EvaluateMomentumDirection();
void   TryOpenMomentumTrade();
void   ApplyTimeoutProtection();
int    CountManagedPositions();
ulong  FindLatestPositionTicket();
void   RegisterPositionContext(const ulong ticket,const ENUM_ORDER_TYPE direction);
ENUM_ORDER_TYPE LookupPositionDirection(const ulong ticket);
void   UnregisterPositionContext(const ulong ticket);
void   MarkTimeoutTicket(const ulong ticket);
bool   ConsumeTimeoutFlag(const ulong ticket);
double GetPipSize();
double NormalizeVolume(const double lots);
double CalculatePositionSize(const ENUM_ORDER_TYPE orderType,const double entryPrice,const double stopPrice);
void   UpdateBacktestTracking();
double CalculateMonths(const datetime start_time,const datetime end_time);

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
  {
   MathSrand((int)TimeLocal());

   g_slotsToFill=(InpMaxConcurrentPositions>0 ? InpMaxConcurrentPositions : 1);
   g_positionCount=0;
   g_timeoutCount=0;
   g_lastEntryTime=0;
   g_testStart=0;
   g_testEnd=0;
   g_totalBars=0;
   g_lastBarTime=0;

   if(!EnsureSymbol())
      return(INIT_FAILED);

   InitializeDirection();

   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetAsyncMode(false);

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Capture the final timestamp when the expert shuts down
   g_testEnd=TimeCurrent();
  }

//+------------------------------------------------------------------+
//| Expert tick handler                                              |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!EnsureSymbol())
      return;

   UpdateBacktestTracking();
   ApplyTimeoutProtection();

   TryOpenMomentumTrade();
  }

//+------------------------------------------------------------------+
//| Trade transaction handler                                        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,const MqlTradeRequest &request,const MqlTradeResult &result)
  {
   if(trans.type!=TRADE_TRANSACTION_DEAL_ADD)
      return;

   if(trans.magic!=(long)InpMagicNumber)
      return;

   if(trans.symbol!=g_symbol)
      return;

   if(trans.entry==DEAL_ENTRY_IN)
     {
      // Register the newly opened position so we can track its direction
      ENUM_ORDER_TYPE dir=(trans.deal_type==DEAL_TYPE_BUY ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
      RegisterPositionContext(trans.position,dir);
      return;
     }

   if(trans.entry!=DEAL_ENTRY_OUT)
      return;

   const ulong positionId=trans.position;
   const ENUM_ORDER_TYPE originalDirection=LookupPositionDirection(positionId);
   const bool timedOut=ConsumeTimeoutFlag(positionId);

   MomentumOutcome outcome=MOMENTUM_NONE;
   if(timedOut)
      outcome=MOMENTUM_TIMEOUT;
   else
     {
      const double netProfit=trans.profit + trans.swap + trans.commission;
      if(netProfit>0.0)
         outcome=MOMENTUM_PROFIT;
      else if(netProfit<0.0)
         outcome=MOMENTUM_LOSS;
      else
         outcome=MOMENTUM_NONE;
     }

   switch(outcome)
     {
      case MOMENTUM_PROFIT:
         // Keep the current direction so we continue the momentum chain
         break;

      case MOMENTUM_LOSS:
         // Reverse on losses to align with the emerging move
         if(originalDirection==ORDER_TYPE_BUY)
            g_currentDirection=ORDER_TYPE_SELL;
         else
            g_currentDirection=ORDER_TYPE_BUY;
         break;

      case MOMENTUM_TIMEOUT:
         // Timeout triggers a fresh momentum scan
         EvaluateMomentumDirection();
         break;

      default:
         // Neutral outcomes fall back to a fresh momentum evaluation
         EvaluateMomentumDirection();
         break;
     }

   UnregisterPositionContext(positionId);

   // Free a slot so that a fresh trade can be submitted on the next tick
   if(g_slotsToFill<InpMaxConcurrentPositions)
      g_slotsToFill++;
  }

//+------------------------------------------------------------------+
//| Ensure the trading symbol is ready                               |
//+------------------------------------------------------------------+
bool EnsureSymbol()
  {
   if(g_symbol!="")
      return(true);

   g_symbol=(InpSymbol=="" ? Symbol() : InpSymbol);
   if(g_symbol=="")
      return(false);

   if(!SymbolSelect(g_symbol,true))
      return(false);

   return(true);
  }

//+------------------------------------------------------------------+
//| Seed the initial trade direction randomly                        |
//+------------------------------------------------------------------+
void InitializeDirection()
  {
   if(MathRand()%2==0)
      g_currentDirection=ORDER_TYPE_BUY;
   else
      g_currentDirection=ORDER_TYPE_SELL;
  }

//+------------------------------------------------------------------+
//| Evaluate momentum from recent price action                       |
//+------------------------------------------------------------------+
void EvaluateMomentumDirection()
  {
   if(!EnsureSymbol())
      return;

   int bullish=0;
   int bearish=0;
   const int barsToCheck=MathMax(1,InpMomentumBars);
   const int availableBars=Bars(g_symbol,_Period);

   for(int index=1; index<=barsToCheck && index<availableBars; ++index)
     {
      const double closeNow=iClose(g_symbol,_Period,index-1);
      const double closePrev=iClose(g_symbol,_Period,index);
      if(closeNow>closePrev)
         bullish++;
      else if(closeNow<closePrev)
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
//| Attempt to open a trade following the selected direction         |
//+------------------------------------------------------------------+
void TryOpenMomentumTrade()
  {
   if(g_slotsToFill<=0)
      return;

   if(CountManagedPositions()>=InpMaxConcurrentPositions)
      return;

   if(InpRiskPercent<=0.0 || InpStopLossPips<=0.0 || InpTakeProfitPips<=0.0)
      return;

   EvaluateMomentumDirection();

   MqlTick tick;
   if(!SymbolInfoTick(g_symbol,tick))
      return;

   const double pipSize=GetPipSize();
   if(pipSize<=0.0)
      return;

   double entryPrice=0.0;
   double stopPrice=0.0;
   double targetPrice=0.0;

   if(g_currentDirection==ORDER_TYPE_BUY)
     {
      entryPrice=tick.ask;
      stopPrice=entryPrice-InpStopLossPips*pipSize;
      targetPrice=entryPrice+InpTakeProfitPips*pipSize;
     }
   else
     {
      entryPrice=tick.bid;
      stopPrice=entryPrice+InpStopLossPips*pipSize;
      targetPrice=entryPrice-InpTakeProfitPips*pipSize;
     }

   const long digits=(long)SymbolInfoInteger(g_symbol,SYMBOL_DIGITS);
   const int priceDigits=(digits>0 ? (int)digits : (int)Digits());
   stopPrice=NormalizeDouble(stopPrice,priceDigits);
   targetPrice=NormalizeDouble(targetPrice,priceDigits);

   const double volume=CalculatePositionSize(g_currentDirection,entryPrice,stopPrice);
   if(volume<=0.0)
      return;

   bool sent=false;
   if(g_currentDirection==ORDER_TYPE_BUY)
      sent=g_trade.Buy(volume,g_symbol,entryPrice,stopPrice,targetPrice);
   else
      sent=g_trade.Sell(volume,g_symbol,entryPrice,stopPrice,targetPrice);

   if(!sent)
      return;

   // Discover the real position ticket so we can track it later
   const ulong positionTicket=FindLatestPositionTicket();
   if(positionTicket!=0)
      RegisterPositionContext(positionTicket,g_currentDirection);

   g_lastEntryTime=TimeCurrent();

   if(g_slotsToFill>0)
      g_slotsToFill--;
  }

//+------------------------------------------------------------------+
//| Enforce the maximum holding time                                 |
//+------------------------------------------------------------------+
void ApplyTimeoutProtection()
  {
   if(InpMaxTimeMinutes<=0)
      return;

   const int total=PositionsTotal();
   if(total<=0)
      return;

   const int timeoutSeconds=InpMaxTimeMinutes*60;
   const datetime now=TimeCurrent();

   for(int index=0; index<total; ++index)
     {
      if(!PositionSelectByIndex(index))
         continue;

      if((long)PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber)
         continue;

      string symbol="";
      PositionGetString(POSITION_SYMBOL,symbol);
      if(symbol!=g_symbol)
         continue;

      const datetime openTime=(datetime)PositionGetInteger(POSITION_TIME);
      if(openTime==0)
         continue;

      if((now-openTime)<timeoutSeconds)
         continue;

      const ulong ticket=(ulong)PositionGetInteger(POSITION_TICKET);
      if(ticket==0)
         continue;

      MarkTimeoutTicket(ticket);
      g_trade.PositionClose(ticket);
     }
  }

//+------------------------------------------------------------------+
//| Count active positions controlled by the expert                  |
//+------------------------------------------------------------------+
int CountManagedPositions()
  {
   int managed=0;
   const int total=PositionsTotal();

   for(int index=0; index<total; ++index)
     {
      if(!PositionSelectByIndex(index))
         continue;

      if((long)PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber)
         continue;

      string symbol="";
      PositionGetString(POSITION_SYMBOL,symbol);
      if(symbol!=g_symbol)
         continue;

      managed++;
     }

   return(managed);
  }

//+------------------------------------------------------------------+
//| Locate the newest open position ticket                           |
//+------------------------------------------------------------------+
ulong FindLatestPositionTicket()
  {
   ulong latestTicket=0;
   datetime latestTime=0;

   const int total=PositionsTotal();
   for(int index=0; index<total; ++index)
     {
      if(!PositionSelectByIndex(index))
         continue;

      if((long)PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber)
         continue;

      string symbol="";
      PositionGetString(POSITION_SYMBOL,symbol);
      if(symbol!=g_symbol)
         continue;

      const datetime openTime=(datetime)PositionGetInteger(POSITION_TIME);
      const ulong ticket=(ulong)PositionGetInteger(POSITION_TICKET);
      if(ticket==0)
         continue;

      if(openTime>=latestTime)
        {
         latestTime=openTime;
         latestTicket=ticket;
        }
     }

   return(latestTicket);
  }

//+------------------------------------------------------------------+
//| Register the position for later direction lookups                |
//+------------------------------------------------------------------+
void RegisterPositionContext(const ulong ticket,const ENUM_ORDER_TYPE direction)
  {
   if(ticket==0)
      return;

   for(int i=0; i<g_positionCount; ++i)
     {
      if(g_positionBook[i].position_id==ticket)
        {
         g_positionBook[i].direction=direction;
         return;
        }
     }

   if(g_positionCount>=MAX_MANAGED_POSITIONS)
      return;

   g_positionBook[g_positionCount].position_id=ticket;
   g_positionBook[g_positionCount].direction=direction;
   g_positionCount++;
  }

//+------------------------------------------------------------------+
//| Retrieve the stored direction for a ticket                       |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE LookupPositionDirection(const ulong ticket)
  {
   for(int i=0; i<g_positionCount; ++i)
     {
      if(g_positionBook[i].position_id==ticket)
         return(g_positionBook[i].direction);
     }

   return(g_currentDirection);
  }

//+------------------------------------------------------------------+
//| Remove a ticket from the tracking table                          |
//+------------------------------------------------------------------+
void UnregisterPositionContext(const ulong ticket)
  {
   for(int i=0; i<g_positionCount; ++i)
     {
      if(g_positionBook[i].position_id==ticket)
        {
         const int lastIndex=g_positionCount-1;
         if(i!=lastIndex)
            g_positionBook[i]=g_positionBook[lastIndex];
         g_positionCount--;
         break;
        }
     }
  }

//+------------------------------------------------------------------+
//| Track tickets closed due to timeout                              |
//+------------------------------------------------------------------+
void MarkTimeoutTicket(const ulong ticket)
  {
   if(ticket==0)
      return;

   for(int i=0; i<g_timeoutCount; ++i)
     {
      if(g_timeoutTickets[i]==ticket)
         return;
     }

   if(g_timeoutCount>=MAX_MANAGED_POSITIONS)
      return;

   g_timeoutTickets[g_timeoutCount]=ticket;
   g_timeoutCount++;
  }

//+------------------------------------------------------------------+
//| Consume timeout flags so they are handled only once              |
//+------------------------------------------------------------------+
bool ConsumeTimeoutFlag(const ulong ticket)
  {
   for(int i=0; i<g_timeoutCount; ++i)
     {
      if(g_timeoutTickets[i]==ticket)
        {
         const int lastIndex=g_timeoutCount-1;
         if(i!=lastIndex)
            g_timeoutTickets[i]=g_timeoutTickets[lastIndex];
         g_timeoutCount--;
         return(true);
        }
     }

   return(false);
  }

//+------------------------------------------------------------------+
//| Determine the pip size for the active symbol                     |
//+------------------------------------------------------------------+
double GetPipSize()
  {
   double point=SymbolInfoDouble(g_symbol,SYMBOL_POINT);
   if(point<=0.0)
      return(0.0);

   const long digitsRaw=SymbolInfoInteger(g_symbol,SYMBOL_DIGITS);
   const int digits=(digitsRaw>0 ? (int)digitsRaw : (int)Digits());

   if(digits==3 || digits==5)
      return(point*10.0);

   return(point);
  }

//+------------------------------------------------------------------+
//| Normalize a requested volume to the symbol trading settings      |
//+------------------------------------------------------------------+
double NormalizeVolume(const double lots)
  {
   double step=SymbolInfoDouble(g_symbol,SYMBOL_VOLUME_STEP);
   double minVolume=SymbolInfoDouble(g_symbol,SYMBOL_VOLUME_MIN);
   double maxVolume=SymbolInfoDouble(g_symbol,SYMBOL_VOLUME_MAX);

   if(step<=0.0)
      step=1.0;
   if(minVolume<=0.0)
      minVolume=step;
   if(maxVolume<minVolume)
      maxVolume=minVolume;

   double normalized=MathFloor(lots/step)*step;
   if(normalized<minVolume)
      normalized=minVolume;
   if(normalized>maxVolume)
      normalized=maxVolume;

   const long volumeDigits=SymbolInfoInteger(g_symbol,SYMBOL_VOLUME_DIGITS);
   const int digits=(volumeDigits>=0 ? (int)volumeDigits : 2);

   return(NormalizeDouble(normalized,digits));
  }

//+------------------------------------------------------------------+
//| Risk-based position sizing                                       |
//+------------------------------------------------------------------+
double CalculatePositionSize(const ENUM_ORDER_TYPE orderType,const double entryPrice,const double stopPrice)
  {
   const double equity=AccountInfoDouble(ACCOUNT_EQUITY);
   const double riskMoney=equity*InpRiskPercent/100.0;
   if(riskMoney<=0.0)
      return(0.0);

   const double tickValue=SymbolInfoDouble(g_symbol,SYMBOL_TRADE_TICK_VALUE);
   const double tickSize=SymbolInfoDouble(g_symbol,SYMBOL_TRADE_TICK_SIZE);
   const double point=SymbolInfoDouble(g_symbol,SYMBOL_POINT);

   if(tickValue<=0.0 || tickSize<=0.0 || point<=0.0)
      return(0.0);

   const double stopDistance=MathAbs(entryPrice-stopPrice);
   if(stopDistance<=0.0)
      return(0.0);

   if((orderType==ORDER_TYPE_BUY && stopPrice>=entryPrice) ||
      (orderType==ORDER_TYPE_SELL && stopPrice<=entryPrice))
      return(0.0);

   const double valuePerPoint=tickValue/tickSize;
   const double stopPoints=stopDistance/point;
   const double riskPerLot=stopPoints*valuePerPoint;
   if(riskPerLot<=0.0)
      return(0.0);

   const double rawLots=riskMoney/riskPerLot;
   return(NormalizeVolume(rawLots));
  }

//+------------------------------------------------------------------+
//| Update testing time and bar counters                             |
//+------------------------------------------------------------------+
void UpdateBacktestTracking()
  {
   const datetime now=TimeCurrent();
   if(g_testStart==0)
      g_testStart=now;
   g_testEnd=now;

   if(!EnsureSymbol())
      return;

   const datetime barTime=iTime(g_symbol,_Period,0);
   if(barTime>0 && barTime!=g_lastBarTime)
     {
      g_totalBars++;
      g_lastBarTime=barTime;
     }
  }

//+------------------------------------------------------------------+
//| Convert elapsed time to months                                   |
//+------------------------------------------------------------------+
double CalculateMonths(const datetime start_time,const datetime end_time)
  {
   if(start_time<=0 || end_time<=start_time)
      return(1.0);

   const double seconds=(double)(end_time-start_time);
   const double months=seconds/(30.4375*24.0*60.0*60.0);

   if(months<=0.0)
      return(1.0);

   return(months);
  }

//+------------------------------------------------------------------+
//| Custom tester optimization objective                             |
//+------------------------------------------------------------------+
double OnTester()
  {
   const double totalTrades=TesterStatistics(STAT_TRADES);
   const double totalProfit=TesterStatistics(STAT_PROFIT);
   const double initialDeposit=TesterStatistics(STAT_INITIAL_DEPOSIT);
   const double drawdown=TesterStatistics(STAT_EQUITY_DDREL_PERCENT)/100.0;

   if(g_testStart==0)
      g_testStart=TimeCurrent();
   if(g_testEnd==0)
      g_testEnd=TimeCurrent();

   double bars=(double)g_totalBars;
   if(bars<=0.0)
      bars=1.0;

   const double tradeDensity=totalTrades/bars;
   const double months=CalculateMonths(g_testStart,g_testEnd);

   double monthlyProfit=0.0;
   if(initialDeposit>0.0 && months>0.0)
      monthlyProfit=(totalProfit/initialDeposit)/months;

   double weightSum=InpWeightTrades+InpWeightProfit+InpWeightDrawdown;
   if(weightSum<=0.0)
      weightSum=100.0;

   const double wt=InpWeightTrades/weightSum;
   const double wp=InpWeightProfit/weightSum;
   const double wd=InpWeightDrawdown/weightSum;

   const double objective=(tradeDensity*wt + monthlyProfit*wp - drawdown*wd)*100.0;
   return(objective);
  }

//+------------------------------------------------------------------+

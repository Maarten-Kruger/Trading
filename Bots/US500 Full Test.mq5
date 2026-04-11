//+------------------------------------------------------------------+
//|                  QuantLab_Executable_Lab_V6.mq5                  |
//|                            Copyright 2026, Trading Lab           |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

//--- ENUMS
enum ENUM_TRIGGER_MODE {
   TRIG_MACD,     // 1. MACD Bullish
   TRIG_SMA2,     // 2. SMA 10 > 20
   TRIG_SMA3,     // 3. SMA 3-Way Stacked
   TRIG_STOCH,    // 4. Stoch Over 50
   TRIG_BANDS,    // 5. Price > Mid Band
   TRIG_RANDOM    // 6. Random (100% Chance)
};

enum ENUM_RISK_MODE {
   RISK_STANDARD,               // 1. Standard (Max Candles + SL)
   RISK_WEEKDAYS_ALL,           // 2. Weekdays: Close ALL Friday 16:00
   RISK_WEEKDAYS_HOLD_NEG,      // 3. Weekdays: Close Winners Friday, Hold Losers
   RISK_WEEKDAYS_HOLD_POS,      // 4. Weekdays: Close Losers Friday, Hold Winners
   RISK_WEEKDAYS_ATR_TRAIL,     // 5. Weekdays: ATR Trailing Stop (+ Close ALL Friday)
   RISK_WEEKDAYS_HEDGE,         // 6. Weekdays: Dynamic MA Hedge (+ Close ALL Friday)
   RISK_WEEKDAYS_HOLD_NEG_DD    // 7. Weekdays: Close Winners Friday, Pause on 5% DD
};
//--- INPUT PARAMETERS
input ENUM_TRIGGER_MODE InpTriggerMode   = TRIG_MACD;           // Trigger Mechanism
input ENUM_RISK_MODE    InpRiskMode      = RISK_WEEKDAYS_HOLD_NEG; // Risk Management Strategy

input double            InpLots          = 0.10;                // Lot Size
input int               InpTargetPips    = 100;                 // Target (Pips)
input int               InpMaxOpenTrades = 5;                   // Max concurrent main trades
input long              InpMagicNum      = 987654;              // EA Magic Number

// Standard Mode Specifics
input int               InpMaxCandles    = 500;                 // Max holding bars (Standard Mode)

// Hedge Mode Specifics
input int               InpHedgePips     = 1000;                // Pips down to trigger Hedge
input int               InpMAPeriod      = 14;                  // MA Period for Hedge

// ATR Trailing Mode Specifics
input int               InpATRPeriod     = 14;                  // ATR Period
input double            InpATRMult       = 2.0;                 // ATR Multiplier for Trail Distance

// Base Filter
input int               InpTrendPeriod   = 200;                 // Downtrend Filter EMA

// Drawdown Pause Mode Specifics
input double            InpMaxDrawdownPercent = 5.0; // Max Drawdown % before pausing

//--- GLOBALS
double PipPoint;
CTrade trade;

// Handles
int hMACD, hSMA10, hSMA20, hSMA50, hTrend, hStoch, hBands, hHedgeMA, hATR;

// Stats
int TotalOpened = 0;
int TotalTPHits = 0;
int TotalClosedByRisk = 0;

//+------------------------------------------------------------------+
//| Helper: Find if a hedge exists for a specific main ticket        |
//+------------------------------------------------------------------+
ulong GetHedgeTicket(ulong main_ticket) {
   string target_comment = "H_" + IntegerToString(main_ticket);
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket)) {
         if(PositionGetInteger(POSITION_MAGIC) == InpMagicNum + 1) {
            string comment = PositionGetString(POSITION_COMMENT);
            if(StringFind(comment, target_comment) >= 0) return ticket;
         }
      }
   }
   return 0;
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   PipPoint = (digits == 3 || digits == 5) ? 10 * point : point;

   trade.SetExpertMagicNumber(InpMagicNum);

   hMACD    = iMACD(_Symbol, _Period, 12, 26, 9, PRICE_CLOSE);
   hSMA10   = iMA(_Symbol, _Period, 10, 0, MODE_SMA, PRICE_CLOSE);
   hSMA20   = iMA(_Symbol, _Period, 20, 0, MODE_SMA, PRICE_CLOSE);
   hSMA50   = iMA(_Symbol, _Period, 50, 0, MODE_SMA, PRICE_CLOSE);
   hTrend   = iMA(_Symbol, _Period, InpTrendPeriod, 0, MODE_EMA, PRICE_CLOSE);
   hStoch   = iStochastic(_Symbol, _Period, 5, 3, 3, MODE_SMA, STO_LOWHIGH);
   hBands   = iBands(_Symbol, _Period, 20, 0, 2.0, PRICE_CLOSE);
   
   if(InpRiskMode == RISK_WEEKDAYS_HEDGE) {
      hHedgeMA = iMA(_Symbol, _Period, InpMAPeriod, 0, MODE_SMA, PRICE_CLOSE);
   }
   if(InpRiskMode == RISK_WEEKDAYS_ATR_TRAIL) {
      hATR = iATR(_Symbol, _Period, InpATRPeriod);
   }

   Print("--- Executable Lab V6 Started ---");
   PrintFormat("Testing: %s with %s", EnumToString(InpTriggerMode), EnumToString(InpRiskMode));

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   double curPrice = iClose(_Symbol, _Period, 0);

   // ==========================================
   // 1. RISK MANAGEMENT MODULE (EXITS & HEDGES)
   // ==========================================
   
   // A. Friday Expirations (All Weekdays Modes)
   bool isFridayClose = (dt.day_of_week == 5 && dt.hour >= 16);
   if(isFridayClose && InpRiskMode != RISK_STANDARD) {
      for(int i = PositionsTotal() - 1; i >= 0; i--) {
         ulong ticket = PositionGetTicket(i);
         if(PositionSelectByTicket(ticket)) {
            long magic = PositionGetInteger(POSITION_MAGIC);
            
            // Only evaluate our Main Trades (and Hedge trades if they exist)
            if(magic == InpMagicNum || magic == (InpMagicNum + 1)) {
               double profit = PositionGetDouble(POSITION_PROFIT);
               bool shouldClose = false;

               // Determine if this specific trade should be closed based on the selected Risk Mode
               if(InpRiskMode == RISK_WEEKDAYS_ALL || InpRiskMode == RISK_WEEKDAYS_HEDGE || InpRiskMode == RISK_WEEKDAYS_ATR_TRAIL) {
                  shouldClose = true; 
               } 
               else if((InpRiskMode == RISK_WEEKDAYS_HOLD_NEG || InpRiskMode == RISK_WEEKDAYS_HOLD_NEG_DD) && profit > 0) {
                  shouldClose = true; // Close the winners, let losers run
               } 
               else if(InpRiskMode == RISK_WEEKDAYS_HOLD_POS && profit < 0) {
                  shouldClose = true; // Close the losers, let winners run
               }

               if(shouldClose) {
                  if(trade.PositionClose(ticket)) {
                     if(magic == InpMagicNum) TotalClosedByRisk++;
                  }
               }
            }
         }
      }
      return; // Skip entry logic during the Friday close window
   }

   // B. Max Candles Expiry (Standard Mode Only)
   if(InpRiskMode == RISK_STANDARD) {
      for(int i = PositionsTotal() - 1; i >= 0; i--) {
         ulong ticket = PositionGetTicket(i);
         if(PositionSelectByTicket(ticket) && PositionGetInteger(POSITION_MAGIC) == InpMagicNum) {
            datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);
            int barsPassed = iBarShift(_Symbol, _Period, openTime);
            if(barsPassed >= InpMaxCandles) {
               if(trade.PositionClose(ticket)) TotalClosedByRisk++;
            }
         }
      }
   }

   // C. Dynamic Hedging Engine
   if(InpRiskMode == RISK_WEEKDAYS_HEDGE) {
      double maVal[]; ArraySetAsSeries(maVal, true);
      if(CopyBuffer(hHedgeMA, 0, 0, 1, maVal) > 0) {
         double currentMA = maVal[0];
         
         for(int i = PositionsTotal() - 1; i >= 0; i--) {
            ulong ticket = PositionGetTicket(i);
            if(PositionSelectByTicket(ticket) && PositionGetInteger(POSITION_MAGIC) == InpMagicNum && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
               
               double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
               double volume = PositionGetDouble(POSITION_VOLUME);
               double hedgeThreshold = openPrice - (InpHedgePips * PipPoint);
               ulong existingHedge = GetHedgeTicket(ticket);
               
               // Trigger Hedge
               if(currentMA <= hedgeThreshold && existingHedge == 0) {
                  trade.SetExpertMagicNumber(InpMagicNum + 1);
                  double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
                  trade.Sell(volume, _Symbol, bid, 0, 0, "H_" + IntegerToString(ticket));
                  trade.SetExpertMagicNumber(InpMagicNum);
               }
               // Remove Hedge
               else if(currentMA > hedgeThreshold && existingHedge > 0) {
                  trade.PositionClose(existingHedge);
               }
            }
         }
      }
   }

   // D. ATR Trailing Stop
   if(InpRiskMode == RISK_WEEKDAYS_ATR_TRAIL) {
      double atrVal[]; ArraySetAsSeries(atrVal, true);
      if(CopyBuffer(hATR, 0, 0, 1, atrVal) > 0) {
         double currentATR = atrVal[0];
         for(int i = PositionsTotal() - 1; i >= 0; i--) {
            ulong ticket = PositionGetTicket(i);
            if(PositionSelectByTicket(ticket) && PositionGetInteger(POSITION_MAGIC) == InpMagicNum) {
               double currentSL = PositionGetDouble(POSITION_SL);
               double currentTP = PositionGetDouble(POSITION_TP);
               double proposedSL = curPrice - (currentATR * InpATRMult);
               
               // Only move the Stop Loss UP (trailing), never down. 
               // Adding a 5 pip buffer to prevent server spam for micro-movements.
               if(proposedSL > currentSL + (5 * PipPoint)) {
                  trade.PositionModify(ticket, proposedSL, currentTP);
               }
            }
         }
      }
   }


   // ==========================================
   // 2. TRIGGER MODULE (ENTRIES)
   // ==========================================
   static datetime lastBar = 0;
   datetime currentBar = iTime(_Symbol, _Period, 0);

   if(currentBar != lastBar) {
      lastBar = currentBar;
      
      // Downtrend Filter
      double ema[2]; CopyBuffer(hTrend, 0, 0, 2, ema);
      if(curPrice < ema[0] && ema[0] < ema[1]) return;

      // Day Filter (Only Standard mode can open on any day)
      if(InpRiskMode != RISK_STANDARD) {
         if(dt.day_of_week != 1 && dt.day_of_week != 2) return;
      }
      
      // Drawdown Pause & Recovery Logic
      static bool isPausedForDD = false;
      if(InpRiskMode == RISK_WEEKDAYS_HOLD_NEG_DD) {
         double balance = AccountInfoDouble(ACCOUNT_BALANCE);
         double equity = AccountInfoDouble(ACCOUNT_EQUITY);
         
         // Calculate floating drawdown percentage
         double ddPercent = (balance > 0) ? ((balance - equity) / balance * 100.0) : 0;
         
         // Lock out new trades if DD exceeds the user-defined threshold
         if(ddPercent >= InpMaxDrawdownPercent) {
            isPausedForDD = true;
         } 
         // Un-pause only when equity fully recovers to balance (0% DD)
         else if(ddPercent <= 0.0) {
            isPausedForDD = false; 
         }
         
         // Abort entry evaluation if currently in a timeout state
         if(isPausedForDD) return; 
      }
      
      // Count Main Trades
      int mainTradesCount = 0;
      for(int i = 0; i < PositionsTotal(); i++) {
         ulong ticket = PositionGetTicket(i);
         if(PositionSelectByTicket(ticket) && PositionGetInteger(POSITION_MAGIC) == InpMagicNum)
            mainTradesCount++;
      }

      // Evaluate Entry
      if(mainTradesCount < InpMaxOpenTrades) {
         bool triggerMet = false;
         
         switch(InpTriggerMode) {
            case TRIG_MACD: {
               double m[1], s[1]; CopyBuffer(hMACD, 0, 0, 1, m); CopyBuffer(hMACD, 1, 0, 1, s);
               triggerMet = (m[0] > s[0]); break;
            }
            case TRIG_SMA2: {
               double f[1], s[1]; CopyBuffer(hSMA10, 0, 0, 1, f); CopyBuffer(hSMA20, 0, 0, 1, s);
               triggerMet = (f[0] > s[0]); break;
            }
            case TRIG_SMA3: {
               double f[1], m[1], s[1]; CopyBuffer(hSMA10, 0, 0, 1, f); CopyBuffer(hSMA20, 0, 0, 1, m); CopyBuffer(hSMA50, 0, 0, 1, s);
               triggerMet = (f[0] > m[0] && m[0] > s[0]); break;
            }
            case TRIG_STOCH: {
               double k[1]; CopyBuffer(hStoch, 0, 0, 1, k);
               triggerMet = (k[0] > 50); break;
            }
            case TRIG_BANDS: {
               double mid[1]; CopyBuffer(hBands, 0, 0, 1, mid);
               triggerMet = (curPrice > mid[0]); break;
            }
            case TRIG_RANDOM: {
               // triggerMet = (MathRand() % 100 < 50); break;
               triggerMet = true; break;
            }
         }

         if(triggerMet) {
            double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            double tp = ask + (InpTargetPips * PipPoint);
            double sl = 0;
            
            // Calculate initial Stop Loss based on Risk Mode
            if(InpRiskMode == RISK_STANDARD) {
               sl = ask - (InpTargetPips * PipPoint);
            } 
            else if(InpRiskMode == RISK_WEEKDAYS_ATR_TRAIL) {
               double atrVal[]; ArraySetAsSeries(atrVal, true);
               if(CopyBuffer(hATR, 0, 0, 1, atrVal) > 0) {
                  sl = ask - (atrVal[0] * InpATRMult); // Initial ATR Stop
               }
            }
            
            if(trade.Buy(InpLots, _Symbol, ask, sl, tp, EnumToString(InpTriggerMode))) {
               TotalOpened++;
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Trade Transaction (Track TP Wins & SL Hits)                      |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result) {
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD) {
      if(HistoryDealSelect(trans.deal)) {
         long magic = HistoryDealGetInteger(trans.deal, DEAL_MAGIC);
         long entry = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
         long reason = HistoryDealGetInteger(trans.deal, DEAL_REASON);
         
         if(magic == InpMagicNum && entry == DEAL_ENTRY_OUT) {
            if(reason == DEAL_REASON_TP) {
               TotalTPHits++;
            }
            else if(reason == DEAL_REASON_SL && (InpRiskMode == RISK_STANDARD || InpRiskMode == RISK_WEEKDAYS_ATR_TRAIL)) {
               TotalClosedByRisk++; // Track Stop Loss hits correctly
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   int completed = TotalTPHits + TotalClosedByRisk;
   double winRate = (completed > 0) ? ((double)TotalTPHits / completed) * 100.0 : 0;

   Print("===============================================");
   PrintFormat("TRIGGER: %s | RISK: %s", EnumToString(InpTriggerMode), EnumToString(InpRiskMode));
   Print("===============================================");
   PrintFormat("Main Trades Opened: %d", TotalOpened);
   PrintFormat("Hits TP (Wins):     %d", TotalTPHits);
   PrintFormat("Closed by Risk:     %d", TotalClosedByRisk);
   PrintFormat("Actual Win Rate:    %.2f%%", winRate);
   Print("===============================================");
}
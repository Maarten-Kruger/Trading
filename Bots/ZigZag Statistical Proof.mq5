//+------------------------------------------------------------------+
//|                                              ZigZag_Stat_Bot.mq5 |
//|                                  Market Structure Tracking Logic |
//+------------------------------------------------------------------+
#property copyright "Your Custom Bot"
#property link      ""
#property version   "1.10"

//--- Input Parameters for the ZigZag Indicator
input int InpDepth     = 12; // ZigZag Depth
input int InpDeviation = 5;  // ZigZag Deviation
input int InpStep      = 3;  // ZigZag Step

//--- Global Variables
int      zigzag_handle;
datetime last_locked_time = 0;

//--- Structure to hold ZigZag point data
struct ZZPoint {
    double   price;
    datetime time;
    bool     isHigh;
};

//--- Structure to track our statistics
struct ScenarioStat {
    string name;
    int    total;
    int    outcome1_count; // HH (if bullish) or LL (if bearish)
    int    outcome2_count; // LH (if bullish) or HL (if bearish)
};

// Arrays to hold the 4 bullish and 4 bearish scenarios
ScenarioStat stats_highs[4];
ScenarioStat stats_lows[4];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    // Load the standard MT5 ZigZag indicator
    zigzag_handle = iCustom(_Symbol, _Period, "Examples\\ZigZag", InpDepth, InpDeviation, InpStep);
    
    if(zigzag_handle == INVALID_HANDLE) {
        Print("Failed to load ZigZag indicator. Error: ", GetLastError());
        return(INIT_FAILED);
    }
    
    // Initialize High (Bullish) Scenario Names
    stats_highs[0].name = "HH -> HL";
    stats_highs[1].name = "HH -> LL";
    stats_highs[2].name = "LH -> HL";
    stats_highs[3].name = "LH -> LL";
    
    // Initialize Low (Bearish) Scenario Names
    stats_lows[0].name = "LL -> LH";
    stats_lows[1].name = "LL -> HH";
    stats_lows[2].name = "HL -> LH";
    stats_lows[3].name = "HL -> HH";
    
    Print("ZigZag Stat Bot Initialized. Tracking 8 Market Structure Scenarios...");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function (Runs at end of backtest)       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    IndicatorRelease(zigzag_handle);
    
    // Print the final statistical summary when the bot stops/backtest ends
    Print("\n=================================================");
    Print("          FINAL ZIGZAG STATISTICS                ");
    Print("=================================================");
    
    Print("--- BULLISH TARGETS (When 'i' formed a High) ---");
    for(int i = 0; i < 4; i++) {
        if(stats_highs[i].total > 0) {
            double pct_hh = ((double)stats_highs[i].outcome1_count / stats_highs[i].total) * 100.0;
            double pct_lh = ((double)stats_highs[i].outcome2_count / stats_highs[i].total) * 100.0;
            PrintFormat("[%s] Total: %d | Resulted in HH: %d (%.1f%%) | Resulted in LH: %d (%.1f%%)", 
                        stats_highs[i].name, stats_highs[i].total, 
                        stats_highs[i].outcome1_count, pct_hh, 
                        stats_highs[i].outcome2_count, pct_lh);
        } else {
            PrintFormat("[%s] Total: 0", stats_highs[i].name);
        }
    }
    
    Print("\n--- BEARISH TARGETS (When 'i' formed a Low) ---");
    for(int i = 0; i < 4; i++) {
        if(stats_lows[i].total > 0) {
            double pct_ll = ((double)stats_lows[i].outcome1_count / stats_lows[i].total) * 100.0;
            double pct_hl = ((double)stats_lows[i].outcome2_count / stats_lows[i].total) * 100.0;
            PrintFormat("[%s] Total: %d | Resulted in LL: %d (%.1f%%) | Resulted in HL: %d (%.1f%%)", 
                        stats_lows[i].name, stats_lows[i].total, 
                        stats_lows[i].outcome1_count, pct_ll, 
                        stats_lows[i].outcome2_count, pct_hl);
        } else {
            PrintFormat("[%s] Total: 0", stats_lows[i].name);
        }
    }
    Print("=================================================\n");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    double zz_data[];
    datetime time_data[];
    
    ArraySetAsSeries(zz_data, true);
    ArraySetAsSeries(time_data, true);
    
    if(CopyBuffer(zigzag_handle, 0, 0, 1000, zz_data) <= 0) return;
    if(CopyTime(_Symbol, _Period, 0, 1000, time_data) <= 0) return;
    
    ZZPoint p[6];
    int found = 0;
    
    for(int i = 0; i < ArraySize(zz_data) && found < 6; i++) {
        if(zz_data[i] > 0.0 && zz_data[i] < EMPTY_VALUE) {
            p[found].price = zz_data[i];
            p[found].time = time_data[i];
            found++;
        }
    }
    
    if(found < 6) return;
    
    for(int i = 0; i < 5; i++) {
        p[i].isHigh = (p[i].price > p[i+1].price);
    }
    p[5].isHigh = !p[4].isHigh; 
    
    if(last_locked_time != p[1].time) {
        if(last_locked_time != 0) {
            AnalyzeStructureAndTrack(p);
        }
        last_locked_time = p[1].time;
    }
}

//+------------------------------------------------------------------+
//| Logic Processing and Tracking for the 8 Scenarios                |
//+------------------------------------------------------------------+
void AnalyzeStructureAndTrack(ZZPoint &p[]) {
    string state_i2 = "";
    string state_i1 = "";
    string state_i  = "";
    int idx = -1;
    
    if(p[1].isHigh) {
        // --- BULLISH TARGETS ---
        state_i2 = (p[3].price > p[5].price) ? "HH" : "LH";
        state_i1 = (p[2].price > p[4].price) ? "HL" : "LL";
        state_i  = (p[1].price > p[3].price) ? "HH" : "LH";
        
        // Find which array index this scenario belongs to
        if(state_i2 == "HH" && state_i1 == "HL") idx = 0;
        else if(state_i2 == "HH" && state_i1 == "LL") idx = 1;
        else if(state_i2 == "LH" && state_i1 == "HL") idx = 2;
        else if(state_i2 == "LH" && state_i1 == "LL") idx = 3;
        
        // Update the statistics array
        if(idx != -1) {
            stats_highs[idx].total++;
            if(state_i == "HH") stats_highs[idx].outcome1_count++;
            else stats_highs[idx].outcome2_count++;
            
            PrintFormat("LOG: Scenario [%s] completed as a [%s]", stats_highs[idx].name, state_i);
        }
        
    } else {
        // --- BEARISH TARGETS ---
        state_i2 = (p[3].price > p[5].price) ? "HL" : "LL";
        state_i1 = (p[2].price > p[4].price) ? "HH" : "LH";
        state_i  = (p[1].price > p[3].price) ? "HL" : "LL";
        
        // Find which array index this scenario belongs to
        if(state_i2 == "LL" && state_i1 == "LH") idx = 0;
        else if(state_i2 == "LL" && state_i1 == "HH") idx = 1;
        else if(state_i2 == "HL" && state_i1 == "LH") idx = 2;
        else if(state_i2 == "HL" && state_i1 == "HH") idx = 3;
        
        // Update the statistics array
        if(idx != -1) {
            stats_lows[idx].total++;
            if(state_i == "LL") stats_lows[idx].outcome1_count++;
            else stats_lows[idx].outcome2_count++;
            
            PrintFormat("LOG: Scenario [%s] completed as a [%s]", stats_lows[idx].name, state_i);
        }
    }
}
//+------------------------------------------------------------------+
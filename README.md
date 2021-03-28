# code
Various coding projects.

1. "Investment Input Data Clean" & "Investment Analysis"

      OR: "Play that funky flow control, new Python coder"
        
        Early project analyzing my personal investments. Worked from poor scans of old 
        documents so had to do a bit of manipulating to get the data into a workable 
        format. Then compared returns to a hypothetical portfolio that bought and 
        sold IVV at the same times and dollar value as the trades that I executed (in 
        order to control for non-investing-decision inflows and outflows). Result: +2%. 
        I wonder if Warren's hiring.
       


2. "On-Chain (+) Analysis"
 
     OR: "How I proved the ML principle 'No Free Lunch"
     
     OR: "The perils of being un-pythonic"
        
        An exercise in apply ML concepts from Data Science for Finance to a non-equity
        investment. Takes on-chain Bitcoininfo via API from Glassnode, adds deviation 
        from average over various periods and applies PCA and Random Forest Regressor. 
        Considers the statistical significance of the generated model (and finds none). 
        Next step will be to look at the various metrics individually and conduct 
        supervised selection for inputs.

const express = require('express');
const { spawn } = require('child_process');
const fs = require('fs');
const app = express();
const port = 3000;

app.use(express.static('public'));
app.use(express.json());

// Step 1: Risk Analysis
app.post('/risk-analysis', (req, res) => {
    const stocks = req.body.stocks;

    if (!stocks || !Array.isArray(stocks)) {
        return res.status(400).json({ error: 'Invalid stock format' });
    }

    const numStocks = stocks.length;
    const stockSymbols = stocks.join(',');

    const riskProcess = spawn('python', [
        'Risk_Reward_Analysis.py',
        numStocks.toString(),
        stockSymbols
    ]);

    riskProcess.stderr.on('data', (data) => {
        console.error(`Risk Analysis error: ${data}`);
    });

    riskProcess.on('close', (code) => {
        if (code !== 0) {
            return res.status(500).json({ error: 'Risk analysis failed' });
        }

        fs.readFile('analysis.txt', 'utf8', (err, data) => {
            if (err) {
                return res.status(500).json({ error: 'Analysis results not found' });
            }

            try {
                const riskData = JSON.parse(data); // Parse as JSON
                res.json({
                    minRisk: riskData.minRisk,
                    maxRisk: riskData.maxRisk,
                    minReturn: riskData.minReturn,
                    maxReturn: riskData.maxReturn,
                });
            } catch (e) {
                res.status(500).json({ error: 'Invalid JSON format in analysis' });
            }
        });
    });
});

// Step 2: Optimization with user-selected risk
app.post('/optimize', (req, res) => {
    let { stocks, userValue, type } = req.body;

    if (!stocks || !Array.isArray(stocks)) {
        return res.status(400).json({ error: 'Invalid input format' });
    }

    const numStocks = stocks.length;
    const stockSymbols = stocks.join(',');
    if (userValue === null) {
        userValue = 0;
    }

    const optimizeProcess = spawn('python', [
        'Optimization_Logic.py',
        numStocks.toString(),
        stockSymbols,
        userValue.toString(),
        type 
    ]);

    optimizeProcess.stderr.on('data', (data) => {
        console.error(`Optimization error: ${data}`);
    });

    optimizeProcess.on('close', (code) => {
        if (code !== 0) {
            return res.status(500).json({ error: 'Optimization failed' });
        }

        fs.readFile('results.txt', 'utf8', (err, data) => {
            if (err) {
                return res.status(500).json({ error: 'Results not found' });
            }

            try {
                const result = JSON.parse(data);
                const response = {
                    ratios: {},
                    return: result.expected_portfolio_return,
                    risk: result.annual_portfolio_variance,
                    ratio: result.expected_ratio,
                };

                stocks.forEach((stock, index) => {
                    response.ratios[stock] = result.optimized_weights[index];
                });

                res.json(response);
            } catch (e) {
                res.status(500).json({ error: 'Invalid results format' });
            }
        });
    });
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});

const express = require('express');
const { spawn } = require('child_process');
const fs = require('fs');
const app = express();
const port = 3000;

app.use(express.static('public'));
app.use(express.json());

app.post('/optimize', (req, res) => {
    const stocks = req.body.stocks;
    
    if (!stocks || !Array.isArray(stocks)) {
        return res.status(400).json({ error: 'Invalid stock format' });
    }

    const numStocks = stocks.length;
    const stockSymbols = stocks.join(',');

    const pythonProcess = spawn('python', [
        'Optimization_Logic.py',
        numStocks.toString(),
        stockSymbols
    ]);

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
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
                    risk: result.annual_portfolio_variance
                };

                // Map weights to stock symbols
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

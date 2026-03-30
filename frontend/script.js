// Wait for DOM to load
document.addEventListener('DOMContentLoaded', () => {

    // --- State ---
    let currentGraphData = null;
    let selectedProblem = "MAXCUT";
    let graphObj = null;

    // --- DOM Elements ---
    const btnGenerate = document.getElementById('btn-generate');
    const btnCompareQaoa = document.getElementById('btn-compare-qaoa');
    const btnCompareSol = document.getElementById('btn-compare-sol');
    
    const nodeCountInput = document.getElementById('node-count');
    const nodeValDisplay = document.getElementById('node-val');
    const problemSelect = document.getElementById('problem-type');
    const graphStats = document.getElementById('graph-stats');
    
    // Results DOM
    const classicTime = document.getElementById('res-classic-time');
    const classicVal = document.getElementById('res-classic-val');
    const classicRaw = document.getElementById('res-classic-raw');
    const spinnerClassic = document.getElementById('spinner-classic');

    const iaTime = document.getElementById('res-ia-time');
    const iaVal = document.getElementById('res-ia-val');
    const iaRaw = document.getElementById('res-ia-raw');
    const spinnerIa = document.getElementById('spinner-ia');

    // --- Events ---
    nodeCountInput.addEventListener('input', (e) => {
        nodeValDisplay.textContent = e.target.value;
    });

    problemSelect.addEventListener('change', (e) => {
        selectedProblem = e.target.value;
    });

    btnGenerate.addEventListener('click', generateGraph);
    btnCompareQaoa.addEventListener('click', runComparisonQAOA);
    btnCompareSol.addEventListener('click', runComparisonSolution);

    // --- Graph Rendering (Force-Graph) ---
    function renderGraph(data) {
        const container = document.getElementById('graph-container');
        // Clear previous
        container.innerHTML = "";
        
        // Initialize ForceGraph
        graphObj = ForceGraph()(container)
            .graphData(data)
            .nodeId('id')
            .nodeLabel('name')
            .nodeColor(() => '#60a5fa') // Accent blue
            .nodeRelSize(8)
            .linkColor(() => 'rgba(255, 255, 255, 0.2)')
            .linkWidth(2)
            .backgroundColor('transparent')
            .width(container.clientWidth)
            .height(container.clientHeight);
            
        // Resize handler
        window.addEventListener('resize', () => {
            if(graphObj) {
                graphObj.width(container.clientWidth).height(container.clientHeight);
            }
        });
    }

    // --- API Calls ---
    
    // 1. Generate Graph
    async function generateGraph() {
        setLoadingState(true, true);
        resetDashboards();
        btnGenerate.disabled = true;
        
        try {
            const reqBody = {
                n_nodes: parseInt(nodeCountInput.value),
                p_edge: 0.5,
                seed: Math.floor(Math.random() * 1000)
            };

            const res = await fetch('/api/generate_graph', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reqBody)
            });

            if (!res.ok) throw new Error("Failed to generate graph");
            
            const data = await res.json();
            currentGraphData = data;
            
            // Visual Updates
            graphStats.textContent = `${data.nodes.length} Qubits | ${data.links.length} Interactions`;
            renderGraph(data);
            
            enableActionButtons(true);
        } catch (err) {
            console.error(err);
            alert("Error generating graph. Make sure the backend is running.");
        } finally {
            setLoadingState(false, true);
            btnGenerate.disabled = false;
        }
    }

    // 2. Comparison: QAOA
    async function runComparisonQAOA() {
        if(!currentGraphData) return alert("Generate a graph first!");
        
        enableActionButtons(false);
        setLoadingState(true, false);
        resetDashboards();
        
        classicRaw.textContent = `> Starting classic Qiskit optimizer (COBYLA)...\n> Max Iterations: 60\n> Please wait...`;
        iaRaw.textContent = `> FastQuantumPredictor initializing...\n> Extracting Centralities + RWPE...`;

        try {
            const res = await fetch('/api/compare_qaoa', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    graph: currentGraphData,
                    problem: selectedProblem
                })
            });

            if (!res.ok) throw new Error("API Error");
            const data = await res.json();
            
            // Populate Classic
            classicTime.textContent = `${data.qiskit.time_sec.toFixed(2)} s`;
            classicVal.textContent = data.qiskit.energy.toFixed(3);
            classicRaw.textContent = `Optimization Complete.\n` + 
                                     `Energy: ${data.qiskit.energy.toFixed(4)}\n` + 
                                     `Gamma: ${JSON.stringify(data.qiskit.gamma)}\n` + 
                                     `Beta: ${JSON.stringify(data.qiskit.beta)}`;

            // Populate IA
            iaTime.textContent = `${data.ia.time_sec.toFixed(3)} s`;
            iaVal.textContent = data.ia.energy.toFixed(3);
            iaRaw.textContent = `Prediction Instantaneous.\n` + 
                                `Energy evaluated via Qiskit Simulator: ${data.ia.energy.toFixed(4)}\n` + 
                                `Gamma: ${JSON.stringify(data.ia.gamma)}\n` + 
                                `Beta: ${JSON.stringify(data.ia.beta)}\n\n` + 
                                `Speedup: ${(data.qiskit.time_sec / data.ia.time_sec).toFixed(0)}x Faster 🚀`;

            colorNodesByResult(null); // Reset colors if previously colored

        } catch (err) {
            console.error(err);
            classicRaw.textContent = "Error executing comparison.";
            iaRaw.textContent = "Error executing comparison.";
        } finally {
            setLoadingState(false, false);
            enableActionButtons(true);
        }
    }

    // 3. Comparison: Exact Solution
    async function runComparisonSolution() {
        if(!currentGraphData) return alert("Generate a graph first!");

        // Prevent freezing the browser with huge brute force
        if(currentGraphData.nodes.length > 20) {
            if(!confirm("Warning: Exact Brute Force on > 20 nodes might take extremely long or crash the backend. Proceed?")) 
                return;
        }

        enableActionButtons(false);
        setLoadingState(true, false);
        resetDashboards();
        
        classicRaw.textContent = `> Launching Brute Force (2^N combinations)...\n> Exploring Hilbert Space...`;
        iaRaw.textContent = `> QuantumGraphModel initializing...\n> Running GAT + Graph Transformer...`;

        try {
            const res = await fetch('/api/compare_solution', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    graph: currentGraphData,
                    problem: selectedProblem
                })
            });

            if (!res.ok) throw new Error("API Error");
            const data = await res.json();
            
            // Populate Classic (Brute Force)
            classicTime.textContent = `${data.exact.time_sec.toFixed(4)} s`;
            classicVal.textContent = data.exact.value;
            classicRaw.textContent = `Ground Truth Reached.\n` + 
                                     `Max Objective: ${data.exact.value}\n` + 
                                     `Bitstring: [${data.exact.solution.join(', ')}]`;

            // Populate IA
            iaTime.textContent = `${data.ia.time_sec.toFixed(4)} s`;
            iaVal.textContent = data.ia.value;
            
            let ratio = data.exact.value > 0 ? (data.ia.value / data.exact.value * 100).toFixed(1) : 0;
            
            iaRaw.textContent = `Transformer Output Layer reached.\n` + 
                                `Objective found: ${data.ia.value}\n` + 
                                `Bitstring: [${data.ia.solution.join(', ')}]\n\n` + 
                                `Quality ratio: ${ratio}% of Ground Truth`;

            // Visually color nodes based on IA solution
            colorNodesByResult(data.ia.solution);

        } catch (err) {
            console.error(err);
            classicRaw.textContent = "Error executing exact comparison.";
            iaRaw.textContent = "Error executing exact comparison.";
        } finally {
            setLoadingState(false, false);
            enableActionButtons(true);
        }
    }

    // --- Utilities ---
    
    function setLoadingState(isLoading, isGraphGen = false) {
        if(isGraphGen) {
            graphStats.textContent = isLoading ? "Generating Physics Topology..." : graphStats.textContent;
            return;
        }
        
        if (isLoading) {
            spinnerClassic.classList.remove('hidden');
            spinnerIa.classList.remove('hidden');
        } else {
            spinnerClassic.classList.add('hidden');
            spinnerIa.classList.add('hidden');
        }
    }

    function enableActionButtons(enable) {
        btnCompareQaoa.disabled = !enable;
        btnCompareSol.disabled = !enable;
    }

    function resetDashboards() {
        classicTime.textContent = "-- s";
        classicVal.textContent = "--";
        classicRaw.textContent = "// Output will appear here...";
        
        iaTime.textContent = "-- s";
        iaVal.textContent = "--";
        iaRaw.textContent = "// Output will appear here...";
    }

    function colorNodesByResult(bitstring) {
        if(!graphObj) return;
        if(!bitstring) {
            // Reset to default
            graphObj.nodeColor(() => '#60a5fa');
            return;
        }

        // Color Partition A (1) with intense green, Partition B (0) with deep blue
        graphObj.nodeColor(node => {
            const val = bitstring[node.id];
            return val === 1 ? '#34d399' : '#1e3a8a'; // Emerald vs deep dark blue
        });
    }

    // Initialize an empty graph on load to show layout
    enableActionButtons(false);
});

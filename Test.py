"""
Test file for BWO and MOBWO algorithms using benchmark functions
Run this file separately from MOBWO.py
"""

import numpy as np
import matplotlib.pyplot as plt
from MOBWO import BWO, MOBWO

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# SINGLE-OBJECTIVE BENCHMARK FUNCTIONS
# ============================================================================

def sphere(x):
    """Sphere function - unimodal, global min: 0 at [0,0,...,0]"""
    return np.sum(x**2)

def rastrigin(x):
    """Rastrigin function - multimodal, global min: 0 at [0,0,...,0]"""
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock(x):
    """Rosenbrock function - unimodal, global min: 0 at [1,1,...,1]"""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley(x):
    """Ackley function - multimodal, global min: 0 at [0,0,...,0]"""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e

def griewank(x):
    """Griewank function - multimodal, global min: 0 at [0,0,...,0]"""
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_part - prod_part + 1

def schwefel(x):
    """Schwefel function - multimodal, global min: 0 at [420.9687,...,420.9687]"""
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def step(x):
    """Step function - unimodal, global min: 0 at [-0.5, -0.5, ..., -0.5]"""
    return np.sum(np.floor(x + 0.5)**2)

# ============================================================================
# MULTI-OBJECTIVE BENCHMARK FUNCTIONS
# ============================================================================

def zdt1_obj1(x):
    """ZDT1 - Objective 1"""
    return x[0]

def zdt1_obj2(x):
    """ZDT1 - Objective 2"""
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    h = 1 - np.sqrt(x[0] / g)
    return g * h

def zdt2_obj1(x):
    """ZDT2 - Objective 1"""
    return x[0]

def zdt2_obj2(x):
    """ZDT2 - Objective 2"""
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    h = 1 - (x[0] / g)**2
    return g * h

def schaffer1_obj1(x):
    """Schaffer function N.1 - Objective 1"""
    return x[0]

def schaffer1_obj2(x):
    """Schaffer function N.1 - Objective 2"""
    return x[1]

def kursawe_obj1(x):
    """Kursawe - Objective 1"""
    n = len(x)
    sum1 = 0
    for i in range(n-1):
        sum1 += -10 * np.exp(-0.2 * np.sqrt(x[i]**2 + x[i+1]**2))
    return sum1

def kursawe_obj2(x):
    """Kursawe - Objective 2"""
    n = len(x)
    sum1 = 0
    for i in range(n):
        sum1 += np.abs(x[i])**0.8 + 5 * np.sin(x[i]**3)
    return sum1

def viennet_obj1(x):
    """Viennet - Objective 1"""
    return x[0]**2 + (x[1] - 1)**2

def viennet_obj2(x):
    """Viennet - Objective 2"""
    return x[0]**2 + (x[1] + 1)**2 + 1

def viennet_obj3(x):
    """Viennet - Objective 3"""
    return (x[0] - 1)**2 + x[1]**2 + 2

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_bwo_single_objective():
    """Test BWO on single-objective benchmark functions"""
    print("\n" + "="*80)
    print("TEST 1: BWO - Single Objective Optimization")
    print("="*80)
    
    # Test configurations
    test_cases = [
        {
            "name": "Sphere",
            "func": sphere,
            "dim": 10,
            "lb": np.full(10, -5.12),
            "ub": np.full(10, 5.12),
            "known_min": 0
        },
        {
            "name": "Rastrigin",
            "func": rastrigin,
            "dim": 10,
            "lb": np.full(10, -5.12),
            "ub": np.full(10, 5.12),
            "known_min": 0
        },
        {
            "name": "Rosenbrock",
            "func": rosenbrock,
            "dim": 5,
            "lb": np.full(5, -5),
            "ub": np.full(5, 10),
            "known_min": 0
        },
        {
            "name": "Ackley",
            "func": ackley,
            "dim": 10,
            "lb": np.full(10, -32),
            "ub": np.full(10, 32),
            "known_min": 0
        },
        {
            "name": "Griewank",
            "func": griewank,
            "dim": 10,
            "lb": np.full(10, -600),
            "ub": np.full(10, 600),
            "known_min": 0
        },
        {
            "name": "Schwefel",
            "func": schwefel,
            "dim": 10,
            "lb": np.full(10, -500),
            "ub": np.full(10, 500),
            "known_min": 0
        },
        {
            "name": "Step",
            "func": step,
            "dim": 10,
            "lb": np.full(10, -100),
            "ub": np.full(10, 100),
            "known_min": 0
        }
    ]
    
    results = []
    convergence_curves = []
    function_names = []
    
    for test in test_cases:
        print(f"\n--- {test['name']} Function ---")
        print(f"Dimensions: {test['dim']}")
        
        # Run BWO
        best_sol, best_fit, conv = BWO(
            objective_func=test["func"],
            dim=test["dim"],
            lb=test["lb"],
            ub=test["ub"],
            pop_size=50,
            max_iter=300,
            procreating_rate=0.8,
            cannibalism_rate=0.4,
            mutation_rate=0.4,
            verbose=False
        )
        
        print(f"Best Fitness: {best_fit:.6e}")
        print(f"Error from optimum: {abs(best_fit - test['known_min']):.6e}")
        
        results.append({
            "name": test["name"],
            "best": best_fit,
            "error": abs(best_fit - test['known_min'])
        })
        
        convergence_curves.append(conv)
        function_names.append(test["name"])
    
    # Create single figure with subplots for all BWO convergence curves
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, (name, conv) in enumerate(zip(function_names, convergence_curves)):
        ax = axes[i]
        ax.semilogy(conv, linewidth=2, color='darkred')
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Best Fitness (log scale)', fontsize=10)
        ax.set_title(f'{name} Function', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f5f5f5')
    
    # Hide unused subplot
    axes[-1].axis('off')
    
    plt.suptitle('BWO Algorithm - Convergence Curves for Benchmark Functions', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Create bar chart for best fitness values
    fig2, ax = plt.subplots(figsize=(12, 6))
    names = [r["name"] for r in results]
    best_values = [r["best"] for r in results]
    errors = [r["error"] for r in results]
    
    x_pos = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, best_values, width, label='Best Fitness', color='darkred', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, errors, width, label='Error from Optimum', color='steelblue', alpha=0.7)
    
    ax.set_xlabel('Benchmark Functions', fontsize=12)
    ax.set_ylabel('Value (log scale)', fontsize=12)
    ax.set_title('BWO Performance on Benchmark Functions', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY - BWO Single Objective Results")
    print("="*80)
    print(f"{'Function':<15} {'Best Fitness':<20} {'Error from Optimum':<20}")
    print("-"*55)
    for r in results:
        print(f"{r['name']:<15} {r['best']:.6e}       {r['error']:.6e}")
    
    return results


def test_mobwo_multi_objective():
    """Test MOBWO on multi-objective benchmark functions"""
    print("\n" + "="*80)
    print("TEST 2: MOBWO - Multi-Objective Optimization")
    print("="*80)
    
    # Test configurations
    test_cases = [
        {
            "name": "ZDT1",
            "funcs": [zdt1_obj1, zdt1_obj2],
            "dim": 10,
            "lb": np.full(10, 0),
            "ub": np.full(10, 1),
            "num_obj": 2
        },
        {
            "name": "ZDT2",
            "funcs": [zdt2_obj1, zdt2_obj2],
            "dim": 10,
            "lb": np.full(10, 0),
            "ub": np.full(10, 1),
            "num_obj": 2
        },
        {
            "name": "Schaffer N.1",
            "funcs": [schaffer1_obj1, schaffer1_obj2],
            "dim": 2,
            "lb": np.full(2, -10),
            "ub": np.full(2, 10),
            "num_obj": 2
        },
        {
            "name": "Kursawe",
            "funcs": [kursawe_obj1, kursawe_obj2],
            "dim": 3,
            "lb": np.full(3, -5),
            "ub": np.full(3, 5),
            "num_obj": 2
        },
        {
            "name": "Viennet (3D)",
            "funcs": [viennet_obj1, viennet_obj2, viennet_obj3],
            "dim": 2,
            "lb": np.full(2, -3),
            "ub": np.full(2, 3),
            "num_obj": 3
        }
    ]
    
    results = []
    pareto_fronts = []
    front_size_histories = []
    test_names = []
    
    for test in test_cases:
        print(f"\n--- {test['name']} ---")
        print(f"Dimensions: {test['dim']}, Objectives: {test['num_obj']}")
        
        # Run MOBWO
        pareto_solutions, pareto_fitness, front_sizes = MOBWO(
            objective_funcs=test["funcs"],
            dim=test["dim"],
            lb=test["lb"],
            ub=test["ub"],
            pop_size=100,
            max_iter=300,
            procreating_rate=0.8,
            cannibalism_rate=0.4,
            mutation_rate=0.4,
            archive_size=100,
            verbose=True
        )
        
        pareto_fitness = np.array(pareto_fitness)
        print(f"Pareto Front Size: {len(pareto_solutions)}")
        
        # Calculate metrics for 2D problems
        if test["num_obj"] == 2 and len(pareto_fitness) > 1:
            # Hypervolume approximation
            ref_point = np.max(pareto_fitness, axis=0) + 0.1
            sorted_idx = np.argsort(pareto_fitness[:, 0])
            sorted_fitness = pareto_fitness[sorted_idx]
            
            hypervolume = 0
            for i in range(len(sorted_fitness) - 1):
                width = abs(sorted_fitness[i+1, 0] - sorted_fitness[i, 0])
                height = abs(ref_point[1] - sorted_fitness[i, 1])
                hypervolume += width * height
            
            # Spread metric
            distances = []
            for i in range(len(pareto_fitness)):
                for j in range(i+1, len(pareto_fitness)):
                    dist = np.linalg.norm(pareto_fitness[i] - pareto_fitness[j])
                    distances.append(dist)
            
            avg_distance = np.mean(distances) if distances else 0
            
            print(f"Hypervolume (approx): {hypervolume:.4f}")
            print(f"Average Distance: {avg_distance:.4f}")
            
            results.append({
                "name": test["name"],
                "pareto_size": len(pareto_solutions),
                "hypervolume": hypervolume,
                "avg_distance": avg_distance
            })
        else:
            results.append({
                "name": test["name"],
                "pareto_size": len(pareto_solutions),
                "hypervolume": None,
                "avg_distance": None
            })
        
        pareto_fronts.append(pareto_fitness)
        front_size_histories.append(front_sizes)
        test_names.append(test["name"])
    
    # Create single figure with subplots for Pareto fronts
    n_tests = len(test_cases)
    n_cols = 3
    n_rows = (n_tests + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (name, pf, test) in enumerate(zip(test_names, pareto_fronts, test_cases)):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        if test["num_obj"] == 2:
            ax.scatter(pf[:, 0], pf[:, 1], c='darkred', marker='o', alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
            ax.set_xlabel('Objective 1 (Minimize)', fontsize=9)
            ax.set_ylabel('Objective 2 (Minimize)', fontsize=9)
        elif test["num_obj"] == 3:
            # For 3D, show 2D projection or simple scatter
            ax.scatter(pf[:, 0], pf[:, 1], c=pf[:, 2], marker='o', alpha=0.7, s=30, cmap='viridis')
            ax.set_xlabel('Objective 1', fontsize=9)
            ax.set_ylabel('Objective 2', fontsize=9)
            # Add colorbar for 3rd objective
            cbar = plt.colorbar(ax.collections[0], ax=ax)
            cbar.set_label('Objective 3', fontsize=8)
        
        ax.set_title(f'{name} (Size: {len(pf)})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f5f5f5')
    
    # Hide unused subplots
    for i in range(len(test_cases), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('MOBWO Algorithm - Pareto Fronts for Benchmark Problems', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Create figure for Pareto front size evolution
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes2 = axes2.reshape(1, -1)
    
    for i, (name, history) in enumerate(zip(test_names, front_size_histories)):
        row = i // n_cols
        col = i % n_cols
        ax = axes2[row, col]
        
        ax.plot(history, linewidth=2, color='steelblue')
        ax.set_xlabel('Iteration', fontsize=9)
        ax.set_ylabel('Pareto Front Size', fontsize=9)
        ax.set_title(f'{name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f5f5f5')
    
    # Hide unused subplots
    for i in range(len(test_cases), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes2[row, col].axis('off')
    
    plt.suptitle('MOBWO Algorithm - Pareto Front Size Evolution', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY - MOBWO Multi-Objective Results")
    print("="*80)
    print(f"{'Function':<15} {'Pareto Size':<15} {'Hypervolume':<15} {'Avg Distance':<15}")
    print("-"*60)
    for r in results:
        hv = f"{r['hypervolume']:.4f}" if r['hypervolume'] else "N/A"
        dist = f"{r['avg_distance']:.4f}" if r['avg_distance'] else "N/A"
        print(f"{r['name']:<15} {r['pareto_size']:<15} {hv:<15} {dist:<15}")
    
    return results


def compare_algorithms():
    """Compare BWO with different parameter settings"""
    print("\n" + "="*80)
    print("TEST 3: Parameter Sensitivity Analysis")
    print("="*80)
    
    # Test function: Sphere
    func = sphere
    dim = 10
    lb = np.full(dim, -5.12)
    ub = np.full(dim, 5.12)
    
    # Different parameter configurations
    configs = [
        {"name": "Default (P=50, PP=0.8, CR=0.4, PM=0.4)", "pop": 50, "pp": 0.8, "cr": 0.4, "pm": 0.4},
        {"name": "Large Population (P=100)", "pop": 100, "pp": 0.8, "cr": 0.4, "pm": 0.4},
        {"name": "High Procreating (PP=0.9)", "pop": 50, "pp": 0.9, "cr": 0.4, "pm": 0.4},
        {"name": "Low Procreating (PP=0.6)", "pop": 50, "pp": 0.6, "cr": 0.4, "pm": 0.4},
        {"name": "High Cannibalism (CR=0.6)", "pop": 50, "pp": 0.8, "cr": 0.6, "pm": 0.4},
        {"name": "Low Cannibalism (CR=0.2)", "pop": 50, "pp": 0.8, "cr": 0.2, "pm": 0.4},
        {"name": "High Mutation (PM=0.6)", "pop": 50, "pp": 0.8, "cr": 0.4, "pm": 0.6},
        {"name": "Low Mutation (PM=0.2)", "pop": 50, "pp": 0.8, "cr": 0.4, "pm": 0.2},
    ]
    
    # Single figure for all configurations
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    
    best_fitness_results = []
    
    for config, color in zip(configs, colors):
        print(f"\n--- {config['name']} ---")
        
        best_sol, best_fit, conv = BWO(
            objective_func=func,
            dim=dim,
            lb=lb,
            ub=ub,
            pop_size=config["pop"],
            max_iter=200,
            procreating_rate=config["pp"],
            cannibalism_rate=config["cr"],
            mutation_rate=config["pm"],
            verbose=False
        )
        
        print(f"Best Fitness: {best_fit:.6e}")
        ax.semilogy(conv, linewidth=2, color=color, label=config['name'])
        best_fitness_results.append(best_fit)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Fitness (log scale)', fontsize=12)
    ax.set_title('BWO Parameter Sensitivity Analysis (Sphere Function)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f5f5f5')
    
    plt.tight_layout()
    plt.show()
    
    # Bar chart for final best fitness
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    names = [c["name"].split('(')[0] for c in configs]
    x_pos = np.arange(len(names))
    
    bars = ax2.bar(x_pos, best_fitness_results, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Parameter Configuration', fontsize=12)
    ax2.set_ylabel('Best Fitness (log scale)', fontsize=12)
    ax2.set_title('BWO Parameter Comparison - Final Best Fitness', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, best_fitness_results):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{value:.2e}', 
                ha='center', va='bottom', fontsize=8, rotation=0)
    
    plt.tight_layout()
    plt.show()


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*80)
    print("BLACK WIDOW OPTIMIZATION ALGORITHM - TEST SUITE")
    print("Based on: Hayyolalam & Pourhaji Kazem (2020)")
    print("="*80)
    
    # Test 1: Single-objective optimization
    bwo_results = test_bwo_single_objective()
    
    # Test 2: Multi-objective optimization
    mobwo_results = test_mobwo_multi_objective()
    
    # Test 3: Parameter sensitivity
    compare_algorithms()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    run_all_tests()
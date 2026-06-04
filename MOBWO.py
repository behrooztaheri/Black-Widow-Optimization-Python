"""
Black Widow Optimization Algorithm (BWO) and Multi-Objective Black Widow Optimization (MOBWO)

Based on the paper:
    "Black Widow Optimization Algorithm: A novel meta-heuristic approach for
    solving engineering optimization problems"
    Authors: Vahideh Hayyolalam, Ali Asghar Pourhaji Kazem
    Published in: Engineering Applications of Artificial Intelligence 87 (2020) 103249

Algorithm Parameters (from Table 2 of the paper):
    pp  = procreating rate : 0.6   (percentage of population used for reproduction)
    CR  = cannibalism rate : 0.44  (controls fraction of offspring that survive)
    PM  = mutation rate    : 0.4   (percentage of individuals that undergo mutation)

Key stages (Section 3 of the paper):
    1. Initial population  - random spiders over search space
    2. Procreate           - crossover using alpha blending (Equation 1)
    3. Cannibalism         - sexual, sibling, and matriphagy elimination
    4. Mutation            - random swap of two elements in selected widows
    5. Update population   - combine survivors and check stop condition
"""

import numpy as np
from typing import Callable, List, Tuple, Optional


# ============================================================================
# BWO – Single-Objective
# ============================================================================

def BWO(objective_func: Callable,
        dim: int,
        lb: np.ndarray,
        ub: np.ndarray,
        pop_size: int = 50,
        max_iter: int = 1000,
        procreating_rate: float = 0.6,
        cannibalism_rate: float = 0.44,
        mutation_rate: float = 0.4,
        verbose: bool = False) -> Tuple[np.ndarray, float, List[float]]:
    """
    Black Widow Optimization Algorithm for single-objective minimization.

    Parameters
    ----------
    objective_func   : callable  – function f(x) -> float to minimize
    dim              : int       – number of problem variables (Nvar)
    lb               : array     – lower bounds, shape (dim,)
    ub               : array     – upper bounds, shape (dim,)
    pop_size         : int       – population size (Npop)
    max_iter         : int       – maximum iterations
    procreating_rate : float     – fraction of population selected for mating (PP)
    cannibalism_rate : float     – fraction of offspring *eliminated* (CR)
    mutation_rate    : float     – fraction of individuals mutated (PM)
    verbose          : bool      – print progress every 100 iterations

    Returns
    -------
    best_solution    : array     – best found solution vector
    best_fitness     : float     – best found objective value
    convergence_curve: list      – best fitness per iteration (length max_iter+1)
    """

    lb = np.asarray(lb, dtype=float).flatten()
    ub = np.asarray(ub, dtype=float).flatten()
    assert lb.shape == (dim,) and ub.shape == (dim,), "lb/ub must match dim"

    # ------------------------------------------------------------------ #
    # 3.1  Initial population                                              #
    # Paper: candidate widow matrix Npop x Nvar, random within bounds     #
    # ------------------------------------------------------------------ #
    population = np.random.uniform(lb, ub, size=(pop_size, dim))
    fitness = np.array([objective_func(ind) for ind in population])

    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = float(fitness[best_idx])
    convergence_curve = [best_fitness]

    # Number of individuals selected for procreation (nr)
    nr = max(2, int(pop_size * procreating_rate))

    for iteration in range(max_iter):

        # ---------------------------------------------------------------- #
        # Step 1 – Select best nr individuals from current population       #
        # Paper: "Select the best nr solutions in pop and save them in pop1"#
        # ---------------------------------------------------------------- #
        sorted_idx = np.argsort(fitness)
        pop1 = population[sorted_idx[:nr]].copy()

        # ---------------------------------------------------------------- #
        # 3.2  Procreate                                                    #
        # Paper Eq. 1:                                                      #
        #   y1 = alpha*x1 + (1-alpha)*x2                                   #
        #   y2 = alpha*x2 + (1-alpha)*x1                                   #
        # Repeated Nvar/2 times; process applied to all nr pairs.          #
        # ---------------------------------------------------------------- #
        pop2_list = []
        pop2_fit_list = []

        for _ in range(nr):
            # Randomly select two parents from pop1
            p_idx = np.random.choice(len(pop1), 2, replace=False)
            x1, x2 = pop1[p_idx[0]], pop1[p_idx[1]]

            # Generate Nvar children (Nvar/2 alpha values, each yields 2 children)
            # Collect all children and their fitness, then keep the best one
            # (sexual cannibalism: male/father is destroyed; sibling cannibalism
            #  keeps only the fittest spiderlings)
            children = []
            n_pairs = max(1, dim // 2)
            for _ in range(n_pairs):
                alpha = np.random.random()
                y1 = alpha * x1 + (1.0 - alpha) * x2
                y2 = alpha * x2 + (1.0 - alpha) * x1
                children.append(np.clip(y1, lb, ub))
                children.append(np.clip(y2, lb, ub))
            if dim % 2 == 1:                       # handle odd dimension
                alpha = np.random.random()
                children.append(np.clip(alpha * x1 + (1.0 - alpha) * x2, lb, ub))

            # Evaluate all children produced by this pair
            child_fitness = np.array([objective_func(c) for c in children])

            # Sibling cannibalism: keep the CR-fraction best children
            n_survive = max(1, int(len(children) * (1.0 - cannibalism_rate)))
            best_child_idx = np.argsort(child_fitness)[:n_survive]
            for ci in best_child_idx:
                pop2_list.append(children[ci])
                pop2_fit_list.append(child_fitness[ci])

        pop2 = np.array(pop2_list) if pop2_list else np.empty((0, dim))
        pop2_fit = np.array(pop2_fit_list)

        # ---------------------------------------------------------------- #
        # 3.4  Mutation                                                     #
        # Paper: randomly select Mutepop individuals from pop1;             #
        # each chosen solution swaps/replaces one random element.           #
        # ---------------------------------------------------------------- #
        nm = max(1, int(pop_size * mutation_rate))
        pop3_list = []
        pop3_fit_list = []

        for _ in range(nm):
            idx = np.random.randint(0, len(pop1))
            mutated = pop1[idx].copy()
            # Randomly replace one gene with a new random value in bounds
            var_idx = np.random.randint(0, dim)
            mutated[var_idx] = np.random.uniform(lb[var_idx], ub[var_idx])
            pop3_list.append(mutated)
            pop3_fit_list.append(objective_func(mutated))

        pop3 = np.array(pop3_list) if pop3_list else np.empty((0, dim))
        pop3_fit = np.array(pop3_fit_list)

        # ---------------------------------------------------------------- #
        # 3.5 / pseudo-code step 16 – Update population                    #
        # new_pop = pop2 + pop3; maintain Npop size by keeping the best.   #
        # ---------------------------------------------------------------- #
        combined_pop = []
        combined_fit = []

        for arr, fit_arr in [(pop2, pop2_fit), (pop3, pop3_fit)]:
            if len(arr) > 0:
                combined_pop.extend(arr)
                combined_fit.extend(fit_arr)

        # If combined is smaller than pop_size, fill with elite from last population
        if len(combined_pop) < pop_size:
            needed = pop_size - len(combined_pop)
            elite_idx = sorted_idx[:needed]
            for i in elite_idx:
                combined_pop.append(population[i])
                combined_fit.append(fitness[i])

        # Keep the best pop_size individuals
        combined_fit = np.array(combined_fit)
        combined_pop = np.array(combined_pop)
        keep_idx = np.argsort(combined_fit)[:pop_size]
        population = combined_pop[keep_idx]
        fitness = combined_fit[keep_idx]

        # Update global best
        cb_idx = np.argmin(fitness)
        if fitness[cb_idx] < best_fitness:
            best_fitness = float(fitness[cb_idx])
            best_solution = population[cb_idx].copy()

        convergence_curve.append(best_fitness)

        if verbose and (iteration + 1) % 100 == 0:
            print(f"  Iter {iteration + 1:>5}/{max_iter}  |  Best: {best_fitness:.6e}")

    return best_solution, best_fitness, convergence_curve


# ============================================================================
# MOBWO – Multi-Objective
# ============================================================================

def MOBWO(objective_funcs: List[Callable],
          dim: int,
          lb: np.ndarray,
          ub: np.ndarray,
          pop_size: int = 100,
          max_iter: int = 500,
          procreating_rate: float = 0.6,
          cannibalism_rate: float = 0.44,
          mutation_rate: float = 0.4,
          archive_size: int = 100,
          verbose: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Multi-Objective Black Widow Optimization Algorithm (MOBWO).

    Extends BWO to multi-objective problems using:
      - Non-dominated (Pareto) sorting for individual ranking
      - Crowding distance for diversity preservation
      - An external archive of non-dominated solutions

    Parameters
    ----------
    objective_funcs  : list of callables  – each f(x) -> float (to minimize)
    dim              : int                – number of decision variables
    lb               : array             – lower bounds, shape (dim,)
    ub               : array             – upper bounds, shape (dim,)
    pop_size         : int               – population size
    max_iter         : int               – maximum iterations
    procreating_rate : float             – fraction used for mating (PP)
    cannibalism_rate : float             – fraction of offspring eliminated (CR)
    mutation_rate    : float             – fraction mutated (PM)
    archive_size     : int               – maximum Pareto archive size
    verbose          : bool              – print progress every 50 iterations

    Returns
    -------
    pareto_solutions  : list of arrays   – decision vectors on Pareto front
    pareto_fitness    : list of arrays   – objective vectors on Pareto front
    front_sizes       : list of int      – archive size after each iteration
    """

    lb = np.asarray(lb, dtype=float).flatten()
    ub = np.asarray(ub, dtype=float).flatten()
    assert lb.shape == (dim,) and ub.shape == (dim,)

    num_obj = len(objective_funcs)
    nr = max(2, int(pop_size * procreating_rate))

    # ------------------------------------------------------------------ #
    # Helper: evaluate a population matrix (n x dim) -> (n x num_obj)    #
    # ------------------------------------------------------------------ #
    def evaluate(pop: np.ndarray) -> np.ndarray:
        n = len(pop)
        out = np.zeros((n, num_obj))
        for i in range(n):
            for j, f in enumerate(objective_funcs):
                out[i, j] = f(pop[i])
        return out

    # ------------------------------------------------------------------ #
    # Helper: does solution a Pareto-dominate b? (minimisation)           #
    # a dominates b  <=>  a <= b in all objectives AND a < b in at least  #
    # one objective.                                                       #
    # ------------------------------------------------------------------ #
    def dominates(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.all(a <= b) and np.any(a < b))

    # ------------------------------------------------------------------ #
    # Non-dominated sort + crowding distance for a fitness matrix          #
    # Returns:                                                             #
    #   pareto_indices : list of int – indices of Pareto-front solutions   #
    #   crowding       : array (n,)  – crowding distance for each solution #
    # ------------------------------------------------------------------ #
    def non_dominated_sort(fit: np.ndarray) -> Tuple[List[int], np.ndarray]:
        n = len(fit)
        if n == 0:
            return [], np.zeros(0)
        if n == 1:
            return [0], np.array([np.inf])

        # Identify non-dominated solutions
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and not dominated[i]:
                    if dominates(fit[j], fit[i]):
                        dominated[i] = True
                        break

        pareto_idx = [i for i in range(n) if not dominated[i]]

        # Crowding distance (computed only for Pareto members)
        crowding = np.zeros(n)
        m = len(pareto_idx)
        if m <= 2:
            for pi in pareto_idx:
                crowding[pi] = np.inf
        else:
            pareto_fit = fit[pareto_idx]          # shape (m, num_obj)
            for obj in range(num_obj):
                sorted_local = np.argsort(pareto_fit[:, obj])  # indices into pareto_idx
                f_min = pareto_fit[sorted_local[0],  obj]
                f_max = pareto_fit[sorted_local[-1], obj]
                # Boundary solutions get infinite crowding
                crowding[pareto_idx[sorted_local[0]]]  = np.inf
                crowding[pareto_idx[sorted_local[-1]]] = np.inf
                if f_max == f_min:
                    continue
                for k in range(1, m - 1):
                    prev_f = pareto_fit[sorted_local[k - 1], obj]
                    next_f = pareto_fit[sorted_local[k + 1], obj]
                    crowding[pareto_idx[sorted_local[k]]] += (next_f - prev_f) / (f_max - f_min)

        return pareto_idx, crowding

    # ------------------------------------------------------------------ #
    # Full non-dominated ranking (all fronts) for a population             #
    # Returns ranks array (lower rank = better front, rank 1 = Pareto)    #
    # ------------------------------------------------------------------ #
    def full_non_dominated_ranking(fit: np.ndarray) -> np.ndarray:
        n = len(fit)
        ranks = np.zeros(n, dtype=int)
        remaining = list(range(n))
        rank = 1
        while remaining:
            # Build sub-fitness for remaining individuals
            sub_fit = fit[remaining]
            sub_pareto, _ = non_dominated_sort(sub_fit)
            # sub_pareto contains indices into `remaining`
            current_front_original = [remaining[i] for i in sub_pareto]
            for idx in current_front_original:
                ranks[idx] = rank
            remaining = [i for i in remaining if i not in set(current_front_original)]
            rank += 1
        return ranks

    # ------------------------------------------------------------------ #
    # Cannibalism for multi-objective (NSGA-II style selection)            #
    # Keeps the best (1-CR) fraction based on Pareto rank + crowding.     #
    # ------------------------------------------------------------------ #
    def cannibalism_mo(pop: np.ndarray, fit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(pop)
        if n <= 1:
            return pop.copy(), fit.copy()

        ranks = full_non_dominated_ranking(fit)
        _, crowding = non_dominated_sort(fit)   # crowding for ALL n solutions

        # Sort: rank ascending (rank 1 first), then crowding descending
        order = sorted(range(n), key=lambda i: (ranks[i], -crowding[i]))

        n_survive = max(1, int(n * (1.0 - cannibalism_rate)))
        survivors = order[:n_survive]

        return pop[survivors].copy(), fit[survivors].copy()

    # ------------------------------------------------------------------ #
    # Update external Pareto archive                                       #
    # ------------------------------------------------------------------ #
    def update_archive(new_pop: np.ndarray,
                       new_fit: np.ndarray,
                       archive_pop: np.ndarray,
                       archive_fit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Merge population into archive and keep only non-dominated solutions."""
        if len(archive_pop) > 0:
            combined_pop = np.vstack([archive_pop, new_pop])
            combined_fit = np.vstack([archive_fit, new_fit])
        else:
            combined_pop = new_pop.copy()
            combined_fit = new_fit.copy()

        pareto_idx, crowding = non_dominated_sort(combined_fit)
        if len(pareto_idx) == 0:
            return np.empty((0, dim)), np.empty((0, num_obj))

        # If archive exceeds capacity, prune by crowding distance
        if len(pareto_idx) > archive_size:
            sorted_by_crowd = sorted(pareto_idx, key=lambda i: crowding[i], reverse=True)
            pareto_idx = sorted_by_crowd[:archive_size]

        return combined_pop[pareto_idx].copy(), combined_fit[pareto_idx].copy()

    # ------------------------------------------------------------------ #
    # 3.1  Initial population                                              #
    # ------------------------------------------------------------------ #
    population = np.random.uniform(lb, ub, size=(pop_size, dim))
    fitness = evaluate(population)                       # (pop_size, num_obj)

    # Initialise archive
    arc_pop, arc_fit = update_archive(population, fitness,
                                      np.empty((0, dim)), np.empty((0, num_obj)))

    front_sizes: List[int] = []

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #
    for iteration in range(max_iter):

        # --- Select pop1 based on crowding distance (diversity pressure) ---
        _, crowding_all = non_dominated_sort(fitness)
        sorted_crowd = np.argsort(-crowding_all)          # high crowding first
        pop1_idx = sorted_crowd[:nr]
        pop1 = population[pop1_idx].copy()

        # ---- Procreate ------------------------------------------------
        pop2_list: List[np.ndarray] = []
        pop2_fit_list: List[np.ndarray] = []

        for _ in range(nr):
            if len(pop1) < 2:
                break
            p_idx = np.random.choice(len(pop1), 2, replace=False)
            x1, x2 = pop1[p_idx[0]], pop1[p_idx[1]]

            # Generate children via alpha-blending (paper Eq. 1)
            children: List[np.ndarray] = []
            n_pairs = max(1, dim // 2)
            for _ in range(n_pairs):
                alpha = np.random.random()
                y1 = alpha * x1 + (1.0 - alpha) * x2
                y2 = alpha * x2 + (1.0 - alpha) * x1
                children.append(np.clip(y1, lb, ub))
                children.append(np.clip(y2, lb, ub))
            if dim % 2 == 1:
                alpha = np.random.random()
                children.append(np.clip(alpha * x1 + (1.0 - alpha) * x2, lb, ub))

            # Evaluate children
            child_arr = np.array(children)                    # (n_children, dim)
            child_fit = evaluate(child_arr)                   # (n_children, num_obj)

            # Sibling cannibalism on offspring using NSGA-II selection
            if len(child_arr) > 1:
                child_arr, child_fit = cannibalism_mo(child_arr, child_fit)

            for ci in range(len(child_arr)):
                pop2_list.append(child_arr[ci])
                pop2_fit_list.append(child_fit[ci])

        if pop2_list:
            pop2 = np.array(pop2_list)                        # (n, dim)
            pop2_fit = np.array(pop2_fit_list)                # (n, num_obj)
            # Apply cannibalism to the full pop2 pool
            pop2, pop2_fit = cannibalism_mo(pop2, pop2_fit)
        else:
            pop2 = np.empty((0, dim))
            pop2_fit = np.empty((0, num_obj))

        # ---- Mutation -------------------------------------------------
        nm = max(1, int(pop_size * mutation_rate))
        pop3_list: List[np.ndarray] = []
        pop3_fit_list: List[np.ndarray] = []

        for _ in range(min(nm, len(pop1))):
            idx = np.random.randint(0, len(pop1))
            mutated = pop1[idx].copy()
            var_idx = np.random.randint(0, dim)
            mutated[var_idx] = np.random.uniform(lb[var_idx], ub[var_idx])
            pop3_list.append(mutated)
            pop3_fit_list.append(evaluate(mutated.reshape(1, -1))[0])

        if pop3_list:
            pop3 = np.array(pop3_list)
            pop3_fit = np.array(pop3_fit_list)
        else:
            pop3 = np.empty((0, dim))
            pop3_fit = np.empty((0, num_obj))

        # ---- Update population -----------------------------------------
        parts_pop = [arr for arr in [pop2, pop3] if len(arr) > 0]
        parts_fit = [arr for arr in [pop2_fit, pop3_fit] if len(arr) > 0]

        if parts_pop:
            combined_pop = np.vstack(parts_pop)
            combined_fit = np.vstack(parts_fit)
        else:
            combined_pop = np.empty((0, dim))
            combined_fit = np.empty((0, num_obj))

        # Fill up to pop_size using elite from current population
        if len(combined_pop) < pop_size:
            needed = pop_size - len(combined_pop)
            # Pick elites by Pareto rank then crowding
            ranks_cur = full_non_dominated_ranking(fitness)
            _, crowd_cur = non_dominated_sort(fitness)
            elite_order = sorted(range(len(population)),
                                 key=lambda i: (ranks_cur[i], -crowd_cur[i]))
            fill_idx = elite_order[:needed]
            combined_pop = np.vstack([combined_pop, population[fill_idx]])
            combined_fit = np.vstack([combined_fit, fitness[fill_idx]])

        # Keep best pop_size by rank + crowding
        if len(combined_pop) > pop_size:
            combined_pop, combined_fit = cannibalism_mo(
                combined_pop, combined_fit)
            # cannibalism already reduces; if still too many, trim further
            if len(combined_pop) > pop_size:
                ranks_c = full_non_dominated_ranking(combined_fit)
                _, crowd_c = non_dominated_sort(combined_fit)
                order_c = sorted(range(len(combined_pop)),
                                 key=lambda i: (ranks_c[i], -crowd_c[i]))
                keep = order_c[:pop_size]
                combined_pop = combined_pop[keep]
                combined_fit = combined_fit[keep]

        population = combined_pop
        fitness = combined_fit

        # ---- Update Pareto archive ------------------------------------
        arc_pop, arc_fit = update_archive(population, fitness, arc_pop, arc_fit)
        front_sizes.append(len(arc_pop))

        if verbose and (iteration + 1) % 50 == 0:
            print(f"  Iter {iteration + 1:>5}/{max_iter}  |  Archive size: {len(arc_pop)}")

    # Return as lists (consistent with original API)
    pareto_solutions = [arc_pop[i] for i in range(len(arc_pop))]
    pareto_fitness   = [arc_fit[i] for i in range(len(arc_pop))]
    return pareto_solutions, pareto_fitness, front_sizes
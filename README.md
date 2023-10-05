# Biologicky inspirované algoritmy

## Quickstart

```bash
python3.11 -m venv .venv
pip install -r requirements.txt 
```

## Základní pojmy

### Evoluce

1. nastavení hyperparametrů
2. generování počáteční populace
3. prvotní vyhodnocení cenové funkce
4. **výběr rodičů** (výběr nejlepších vede k degradaci!)
5. vytvoření potomků
6. **mutace** potomků (s mírou, např. vhodný parametr $\sigma$ u $N(\mu, \sigma) $)
7. vyhodnocení cenové funkce
8. **elitismus** - výběr nejlepších jednotlivců, obecně z potomků i rodičů
9. **nová populace**
10. nahrazení staré populace novou
11. goto 4 (jedna iterace se nazývá **generace**)

### Strategie

- point strategy (po jednotlivých bodech)
- population strategy (po generacích)

### Optimalizační algoritmy a heuristiky

Můžou vyřešit "black-box" problém, který se nechová podle známého matematického problému. 

- **deterministické** - gradient descent
- **enumerativní** - brute force
- **stochastické**
  - např. Hill Climber, Tabu Search
  - pomalé
  - založené na náhodě
  - vhodné pouze pro malé $\Omega$
- **kombinované** (mixed) - kombinace deterministických a stochastických algoritmů

### No Free Lunch Theorem (NFLT)

- neexistuje algoritmus, který by fungoval na všechny problémy

## Blind Search

- generuju náhodné řešení z prostoru všech řešení

## Hill Climber (Horolezec)

1. nejlepší řešení $\leftarrow$ náhodné řešení z prostoru všech řešení
2. generuju jednotlivce z okolí nejlepšího řešení pomocí normálního rozdělení (se zvoleným parametrem $\sigma$)

## Tabu Search

- podobné jako horolezec, jen při generování jednotlivců (pomocí normálního rozdělění) je vynuceno, aby se neopakovaly předchozí řešení (pomocí fronty a kontroly "`solution is in tabu_queue`")

## Ant Colony Optimization

- vhodný pro kombinatorické problémy

## Simulované žíhání (Simulated Annealing)

## Genetické algoritmy

- **roulette wheel selection**

## Particle Swarm

## Differential Evolition

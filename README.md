# Biologicky inspirované algoritmy

## Quickstart

```bash
python3.11 -m venv .venv
pip install -r requirements.txt 
```

## Základní pojmy

- pro srovnávání algoritmů je vhodná reference **počet ohodnocení cenové funkce** (např. graf počet ohodnocení vs. funkční hodnota)

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

### Evoluční algoritmy

- jednoduchost
- použití decimálních čísel, resp. jejich binárního zápisu
  - využití [Grayova kódu](https://en.wikipedia.org/wiki/Gray_code), kde se každé dvě následující čísla liší pouze jedním bitem

#### Binary-Reflected Gray Code

- reflect-and-prefix method.

![Screenshot from 2023-10-12 09-48-34](https://github.com/matejfric/2023BIA/assets/95862670/d6960193-161a-46d9-83a5-460b4eb25dbe)
 
### Populace 

- každá populace je definována vzorem / zástupcem (specimen), např. $Specimen=( (Float,[Lo,Hi]), (Int,[Lo,Hi]), (Short,[Lo,Hi]) )$
- Co, když je jedinec vygenerován mimo přípustnou množinu?
  - přesun na hranici $\rightarrow$ hromadění jedinců na hranici
  - generace nového jedince, dokud nesplňuje požadavky
  - "pohyb po kouli", $\texttt{if } x>x_{max}: \Delta x = | x_{max} - x | \Rightarrow x = x_{min} + \Delta x$
 
### Testovací funkce

- často mají globální extrém ve "stejném" bodě nehledě na dimenzi (např. Schwefel - $f(\mathbf{x}^{\star})=\mathbf{o}$, $\mathbf{x}^{\star}=(420.97,..., 420.97) $) 

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

1. Nastavení hyperparametrů teploty $T_0 = 1000$, $T_{min} = 0$ a $T_{step} = 10$.
2. Dokud $T > T_{min}$:
   1. Vytvoř jedince $I$ v **přípustné množině** všech řešení $\Omega$ (search space).
   2. Vyhodnoť cenovou funkci $f(I),I\in\Omega$.
   3. Výpočet $\Delta_f = f(I) - f_{best}$.
   4. **Pokud** je nové řešení lepší (tzn. pro minimalizaci $\Delta_f < 0$), aktualizuj řešení.
   5. **Jinak** přijmi nové (horší) řešení, pokud: $r < e^{\frac{\Delta_f}{T}}$, kde $r=\text{random}([0,1))$.
   6. Snížení teploty $T = T - T_{step}$.

Poznámka k výrazu $r < e^{\frac{\Delta_f}{T}}$. Pokud je teplota $T$ vysoká, tak se tento výraz blíží k jedné a je tedy velmi pravděpodobné, že bude přijato nové  řešení (i pokud je horší). Se snižující teplotou se tato pravděpodobnost snižuje.

## Genetické algoritmy

- **roulette wheel selection**
- mutují se bity binárního zápisu decimálního čísla

## Particle Swarm

## Differential Evolution



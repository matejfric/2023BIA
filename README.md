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
6. **mutace** potomků (s mírou, např. vhodný parametr $ \sigma $ u $ \mathcal{N}(\mu, \sigma) $)
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

#### Omezení na argumenty cenové funkce, penalizace, kritické situace

- TODO

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

- tutoriál od [The Hebrew University of Jerusalem](https://www.cs.huji.ac.il/~ai/projects/old/tsp2.pdf)
- založeny na principech evoluce v přírodě - C. R. Darwin a G. J. Mendel
- mutují se bity binárního zápisu decimálního čísla
- používá se *fitness* funkce $F$ normalizovaná do intervalu $[0,1]$

Obecný převod cenové funkce na fitness:

$$
F(i)=\frac{F_{max}-F_{min}}{f_{min}-f_{max}}f(i)+\frac{f_{min}F_{min}-f_{max}F_{max}}{f_{min}-f_{max}}
$$

Na intervalu $[0,1]$:

$$
F(i)=\frac{f(i)-f_{max}}{f_{min}-f_{max}}
$$

Na intervalu $[0,1]$ s "ošetřením $\infty$" ($\varepsilon=0.01$):

$$
F(i)=\frac{(1-\varepsilon)f(i)+f_{min}\varepsilon-f_{max}}{f_{min}-f_{max}}
$$

### Výběr rodičů (selection)

During each successive generation a proportion of the existing population is selected to breed a new generation. Individual solutions are selected through a stochastic fitness-based process, where the requirement is that fitter solutions (as measured by a fitness function) are typically more likely to be selected.

Rodiče se výbírají stochastickým procesem založeným na *fitness* funkci.

1. Roulette Wheel Selection
   - seřadím jedince podle fitness
   - vygeneruju náhodné číslo $r\in[0,1)$
   - vyberu prvního jedince, který má fitness větší než $r$
   - nefunguje dobře pro velké rozdíly mezi rodiči, proto se používá jednoduchá korekce - *Rank Selection*
2. Rank selection
   - jedincům přiřadím hodnoty $1,2,3,...$ podle fitness (nejhorší jedinec 1), odpovídá to "velikosti úseček"
![Rank selection](https://github.com/matejfric/2023BIA/assets/95862670/1cb8457b-b26a-4916-98c1-4d80a8b43c94)

Rank Selection (Roulette Wheel Selection)

![Roulette Wheel Selection](https://github.com/matejfric/2023BIA/assets/95862670/11066f0e-0c14-4f45-b22e-aa8a750fbee7)

$$\mathcal{P}(X=i)=\frac{fitness(i)}{\sum\limits_j fitness(j)}$$

```python
def pick_one(population: list):
    idx = 0
    r = np.random()
    while r > 0:
        r -= population[idx].prob
        idx += 1
    return population[idx - 1]
```

3. Tournament Selection

```pseudocode
choose k (the tournament size) individuals from the population at random
choose the best individual from the tournament with probability p
choose the 2nd best individual with probability p*(1-p)
choose the 3rd best individual with probability p*((1-p)^2)
and so on
```

### Generování potomků - křížení (crossover)

- single point crossover

![single point crossover](https://github.com/matejfric/2023BIA/assets/95862670/fc003100-cf2a-4af4-8ab9-ba71b4dd4c9a)
![single point crossover2](https://github.com/matejfric/2023BIA/assets/95862670/8407011d-a3a6-4de3-9a80-83a31a591d72)

- n-point crossover
  
![n-point crossover](https://github.com/matejfric/2023BIA/assets/95862670/6177219a-4639-4070-b93e-7640388a3cb6)

- uniform crossover (uniformní křížení - každý gen (bit) je náhodně vybrán z jednoho z odpovídajících genů rodičovských chromozomů)
  
![uniform crossover](https://github.com/matejfric/2023BIA/assets/95862670/7e2ac0e5-2efe-47f8-ae62-22646f229083)

U **TSP** vedou předchozí metody k nevalidní konfiguraci.

### Mutace

- TODO

## Diferenciální evoluce (Differential Evolution)

- jeden z dnešních nejlepších evolučních algoritmů
  - ale NFLT: mravenci jsou lepší na kombinatorické výpočty
- vychází z genetických algoritmů

### Pseudokód

1. vygeneruj $NP\in\mathbb{N}$ jedinců počáteční populace
2. pro $G$ generací opakuj:
   1. zkopíruj původní populaci
   2. pro každého jedince předchozí populace - `for parent in population:`
      1. náhodně vyber tři jedince $x_1,x_2,x_3$
      2. jedinci/rodiče z předchozího kroku se podílejí na tvorbě jednoho nového potomka $\boxed{v=(x_1-x_2)*F+x_3}$, kde $F\in[0,2]$ je *mutační konstanta* a $v$ je tzv. *mutační vektor*
      3. proveď *křížení* - pro každý prvek `trial_vector = np.zeros(D)`: pokud je pravděpodobnost (`np.random.uniform()`$\in[0,1)$) menší než $CR\in [0,1]$ (crossover rate), přiřaď do `trial_vector` prvek z mutačního vektoru $v$, jinak zachovej parametr z rodiče (`trial_vector[i] = parent[i]`)
      4. vyhodnoť fitness, pokud je lepší, než aktualní, tak přidej vytvořeného potomka (`trial_vector`) do nové populace a aktualizuj řešení
   3. nahraď starou populaci novou

### Parametry

- $CR\in [0,1]$ - Crossover Rate (doporučení: 0.8 - 0.9)
- $D$ - dimenze
- $NP\in[10D,100D]$ - velikost populace
- $F\in[0,2]$ - mutační konstanta - zkrátí nebo natáhne vektor (0.8)
- $G>0$ - generace

## Particle Swarm

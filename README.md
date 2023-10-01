# Biologicky inspirované algoritmy

## Quickstart

```bash
python3.11 -m venv .venv
pip install -r requirements.txt 
```

## Blind Search

- generuju náhodné řešení z prostoru všech řešení

## Hill Climber (Horolezec)

1. nejlepší řešení $\leftarrow$ náhodné řešení z prostoru všech řešení
2. generuju jednotlivce z okolí nejlepšího řešení pomocí normálního rozdělení (se zvoleným parametrem $\sigma$)

## Tabu Search

- podobné jako horolezec, jen při generování jednotlivců (pomocí normálního rozdělění) je vynuceno, aby se neopakovaly předchozí řešení (pomocí fronty a kontroly "`solution is in tabu_queue`")
  
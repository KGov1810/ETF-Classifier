"""Builds the example data of data/examples/ from FinanceDatabase (MIT licence, github.com/JerBouma/FinanceDatabase).

Real: ticker, name, description and classification ("asset class – segment") of US-listed ETFs.
FICTITIOUS: the composition. The open database has no holdings, so plausible holdings are drawn from a pool of
real securities per classification, with random weights. They are realistic, not the ETFs' actual holdings.

Outputs (UTF-8, ';' separator, input format of the project):
- etfs_train.csv            reference dataset (Classification filled)
- etfs_to_check.csv         ETFs never seen in training; Classification sometimes right, sometimes wrong, empty or
                            unknown: what the model must check and reclassify
- etfs_to_check_answers.csv answer key (true classification and what was changed)

Usage (download compression/etfs.bz2 from the FinanceDatabase repository first):
    python scripts/make_example_data.py --source etfs.bz2 --output data/examples
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
MIN_PER_CLASS, MAX_PER_CLASS = 30, 150
CHECK_SHARE = 0.15             # share of each classification held out for the file to check
NO_DESCRIPTION_SHARE = 0.05    # rows whose description is blanked (composition only), as in a real file

EQUITY_POOLS = {
    "Equities – Developed Markets": "AAPL MSFT NVDA AMZN GOOGL META BRK.B JPM V UNH XOM JNJ PG MA HD AVGO LLY "
                                    "COST ASML NESN NOVO SAP TM SHEL AZN",
    "Equities – Large Cap": "AAPL MSFT NVDA AMZN GOOGL META BRK.B JPM TSLA LLY AVGO V WMT XOM UNH",
    "Equities – Mid Cap": "DECK WSM EME CSL RS LII BLDR TOL WSO FIX CASY OC RPM GGG NBIX",
    "Equities – Small Cap": "SMCI FN ELF CVLT AIT SPSC MLI ANF SFM ONTO IBP BCC CRS FSS MOD",
    "Equities – Growth": "NVDA MSFT AAPL AMZN META TSLA AVGO NFLX CRM ADBE NOW INTU UBER ISRG",
    "Equities – Value": "BRK.B JPM XOM JNJ CVX PG WFC BAC CSCO VZ PFE KO CMCSA MRK",
    "Equities – Factors": "PG KO PEP WMT MRK ABBV MCD VZ TXN CSCO MSFT LLY COST RSG WM",
    "Equities – Emerging Markets": "TSM 005930 TCEHY BABA RELIANCE INFY HDB PDD VALE ITUB 2330 NPN MELI 1299",
    "Equities – Allocation": "IVV AGG VEA IEFA BND VWO TLT SPY IEMG LQD VTI SCHD",
    "Information Technology – Developed Markets": "AAPL MSFT NVDA AVGO ORCL CRM ADBE AMD CSCO ACN IBM QCOM INTU "
                                                  "TXN NOW",
    "Health Care – Developed Markets": "LLY UNH JNJ ABBV MRK TMO ABT ISRG AMGN DHR PFE BSX VRTX GILD",
    "Real Estate – REITs": "PLD AMT EQIX WELL SPG PSA O DLR CCI VICI EXR AVB",
    "Energy – Developed Markets": "XOM CVX COP EOG SLB MPC PSX OXY WMB KMI VLO HES",
    "Industrials – Developed Markets": "GE CAT RTX UNP HON BA LMT DE ETN UPS ADP WM",
    "Materials – Developed Markets": "LIN SHW APD ECL FCX NEM DD NUE DOW VMC MLM CTVA",
    "Alternatives – Alternative": "GC_FUT CL_FUT VIX_FUT SPX_PUT SPX_CALL USD_CASH BTC TLT_PUT EUR_FWD JPY_FWD",
}
CORPORATE_ISSUERS = "AAPL JPM BAC MSFT T VZ AMZN GS C WFC CVS ABBV ORCL CMCSA PFE MS HSBC BA".split()
HIGH_YIELD_ISSUERS = "CCO TDG MEDLINE CCL UAL DISH TEVA F CHTR NCLH HUB QUIKHOLD ATHENAHEALTH".split()
MUNI_ISSUERS = ["NY ST DORM AUTH", "CA ST GO", "NYC TFA", "TX ST TRANS COMM", "IL ST GO", "NJ ECON DEV AUTH",
                "MA ST GO", "LA DEPT WTR PWR", "PUERTO RICO COFINA", "WA ST GO"]
NOISE = "BRK.B JPM CASH_USD MSFT AAPL VZ T XOM".split()  # common holdings across many funds
# Real equity funds overlap heavily: most of them hold the same mega caps
MEGA_CAPS = "AAPL MSFT NVDA AMZN GOOGL META BRK.B JPM AVGO LLY TSLA V XOM UNH MA PG JNJ HD COST WMT".split()
OWN_STYLE = ("Equities – Emerging Markets", "Equities – Allocation", "Alternatives – Alternative",
             "Equities – Mid Cap", "Equities – Small Cap")  # little overlap with the mega caps


def _weights(rng, n: int, coverage: float) -> list[float]:
    raw = np.sort(rng.dirichlet(np.ones(n) * 1.5))[::-1] * coverage
    return [round(float(w), 2) for w in raw]


def fictitious_composition(classification: str, rng) -> str:
    """Plausible top holdings for a classification (real security tickers, random weights)."""
    if classification in EQUITY_POOLS:
        pool = EQUITY_POOLS[classification].split()
        n = int(rng.integers(6, min(12, len(pool)) + 1))
        if classification in OWN_STYLE:
            n_own = n
        else:  # between a third and two thirds of the holdings are the shared mega caps
            n_own = max(2, int(round(n * rng.uniform(0.35, 0.65))))
        tickers = [str(t) for t in rng.choice(pool, size=n_own, replace=False)]
        mega = [t for t in MEGA_CAPS if t not in tickers]
        tickers += [str(t) for t in rng.choice(mega, size=n - n_own, replace=False)]
        tickers += [str(t) for t in rng.choice(NOISE, size=int(rng.integers(0, 3)), replace=False)
                    if t not in tickers]
        coverage = float(rng.uniform(25, 95))
    else:
        n = int(rng.integers(8, 15))
        coverage = float(rng.uniform(8, 35))  # bond funds: many lines, the top holdings weigh little
        segment = classification.split(" – ")[1]
        years = rng.integers(2026, 2055, size=n)
        coupons = rng.choice([1.5, 2.25, 3.0, 3.875, 4.25, 4.5, 5.0, 5.75, 6.5, 7.25], size=n)
        if segment == "Treasury Bonds":
            tickers = [f"UST {c} {y}" for c, y in zip(coupons, years)]
        elif segment == "Inflation-Protected Securities":
            tickers = [f"TII {round(c / 4, 3)} {y}" for c, y in zip(coupons, years)]
        elif segment == "Municipal Bonds":
            tickers = [f"{rng.choice(MUNI_ISSUERS)} {c} {y}" for c, y in zip(coupons, years)]
        elif segment == "High Yield Bonds":
            tickers = [f"{rng.choice(HIGH_YIELD_ISSUERS)} {c + 2} {y}" for c, y in zip(coupons, years)]
        elif segment == "Blend":
            kinds = ["UST", "FNMA 30YR", "FHLMC 15YR", *CORPORATE_ISSUERS[:6]]
            tickers = [f"{rng.choice(kinds)} {c} {y}" for c, y in zip(coupons, years)]
        else:  # Corporate Bonds, Investment Grade Bonds
            tickers = [f"{rng.choice(CORPORATE_ISSUERS)} {c} {y}" for c, y in zip(coupons, years)]
        tickers = list(dict.fromkeys(tickers))
    return str([{str(t): w} for t, w in zip(tickers, _weights(rng, len(tickers), coverage))])


def load_real_etfs(source: Path) -> pd.DataFrame:
    """US-listed, active, non-leveraged ETFs with an informative description and an established classification."""
    df = pd.read_csv(source, compression="bz2")
    keep = (df.summary.notna() & df.category_group.notna() & df.category.notna() & ~df.delisted.fillna(False)
            & (df.currency == "USD") & ~df.symbol.str.contains(".", regex=False)
            & (df.category_group != "Derivatives")
            & ~df.summary.str.contains("is an exchange-traded fund listed on", regex=False)
            & (df.summary.str.len() > 120))
    df = df[keep].drop_duplicates("name").drop_duplicates("symbol")
    df["Classification"] = df.category_group + " – " + df.category
    df["summary"] = [" ".join(text.split()) for text in df.summary]  # one line per ETF
    counts = df.Classification.value_counts()
    df = df[df.Classification.isin(counts[counts >= MIN_PER_CLASS].index)]
    df = df.sample(frac=1, random_state=SEED).groupby("Classification").head(MAX_PER_CLASS)
    return df.rename(columns={"symbol": "Ticker", "name": "Nom", "summary": "Description"})[
        ["Ticker", "Nom", "Description", "Classification"]].reset_index(drop=True)


# Plausible mistakes a person could make: a neighbouring classification
CONFUSIONS = {
    "Equities – Growth": "Equities – Large Cap", "Equities – Large Cap": "Equities – Developed Markets",
    "Equities – Value": "Equities – Factors", "Equities – Factors": "Equities – Developed Markets",
    "Equities – Mid Cap": "Equities – Small Cap", "Equities – Small Cap": "Equities – Mid Cap",
    "Fixed Income – Corporate Bonds": "Fixed Income – Investment Grade Bonds",
    "Fixed Income – Investment Grade Bonds": "Fixed Income – Corporate Bonds",
    "Fixed Income – High Yield Bonds": "Fixed Income – Corporate Bonds",
    "Fixed Income – Treasury Bonds": "Fixed Income – Inflation-Protected Securities",
    "Fixed Income – Inflation-Protected Securities": "Fixed Income – Treasury Bonds",
    "Information Technology – Developed Markets": "Equities – Growth",
    "Health Care – Developed Markets": "Equities – Developed Markets",
    "Real Estate – REITs": "Equities – Value", "Energy – Developed Markets": "Materials – Developed Markets",
    "Materials – Developed Markets": "Industrials – Developed Markets",
}
UNKNOWN_LABELS = ["Equity – Developed Market", "Thematic", "Bonds"]  # misspelt or not in the grid


def make_check_file(held_out: pd.DataFrame, classes: list[str], rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Established classification of the held-out ETFs: 55 % right, 30 % wrong, 10 % empty, 5 % unknown."""
    given, change = [], []
    for true in held_out.Classification:
        draw = rng.random()
        if draw < 0.55:
            given.append(true), change.append("unchanged")
        elif draw < 0.85:
            wrong = CONFUSIONS.get(true) or rng.choice([c for c in classes if c != true])
            given.append(wrong), change.append("wrong classification")
        elif draw < 0.95:
            given.append(""), change.append("no classification")
        else:
            given.append(str(rng.choice(UNKNOWN_LABELS))), change.append("unknown classification")
    to_check = held_out.assign(Classification=given)
    answers = pd.DataFrame({"Ticker": held_out.Ticker, "Nom": held_out.Nom,
                            "TrueClassification": held_out.Classification,
                            "GivenClassification": given, "Change": change})
    return to_check, answers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the example data of data/examples/")
    parser.add_argument("--source", required=True, help="etfs.bz2 of FinanceDatabase")
    parser.add_argument("--output", default="data/examples")
    args = parser.parse_args(argv)
    rng = np.random.default_rng(SEED)

    etfs = load_real_etfs(Path(args.source))
    etfs["Composition"] = [fictitious_composition(c, rng) for c in etfs.Classification]
    blank = rng.random(len(etfs)) < NO_DESCRIPTION_SHARE
    etfs.loc[blank, "Description"] = ""
    held = etfs.groupby("Classification").sample(frac=CHECK_SHARE, random_state=SEED)
    train = etfs.drop(held.index)
    to_check, answers = make_check_file(held.reset_index(drop=True), sorted(etfs.Classification.unique()), rng)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    columns = ["Ticker", "Nom", "Description", "Classification", "Composition"]
    train[columns].to_csv(output / "etfs_train.csv", sep=";", index=False, encoding="utf-8")
    to_check[columns].to_csv(output / "etfs_to_check.csv", sep=";", index=False, encoding="utf-8")
    answers.to_csv(output / "etfs_to_check_answers.csv", sep=";", index=False, encoding="utf-8")
    print(f"{len(train)} training ETFs, {len(to_check)} ETFs to check, "
          f"{etfs.Classification.nunique()} classifications → {output}")
    print(answers.Change.value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

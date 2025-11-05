from wiki_scraper import scraper_bot

if __name__ == "__main__":
    query = "Large language model"
    results = scraper_bot(
        query,
    )

    print(results[0]["title"])
    print(results[0]["content"][:500])  # Print first 500 characters of the content

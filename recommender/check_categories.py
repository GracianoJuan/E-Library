import pandas as pd

df = pd.read_csv('dataset/amazon_books.csv')
print('Categories:')
categories = df['category'].unique()
for i, cat in enumerate(categories[:15]):
    print(f"{i+1}. {cat}")
print(f'\nTotal unique categories: {len(categories)}')

# Show sample books per category
print("\n\nSample books per category:")
for cat in categories[:5]:
    books = df[df['category'] == cat]['title'].head(3).tolist()
    print(f"\n{cat}:")
    for b in books:
        print(f"  - {b}")

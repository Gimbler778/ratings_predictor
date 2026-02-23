import pandas as pd

df = pd.read_csv('Animes.csv')
df.columns = df.columns.str.strip().str.lower().str.replace('[^0-9a-zA-Z]+', '_', regex=True).str.strip('_')

print('Sample image URLs:')
for idx, row in df.head(5).iterrows():
    print(f"{row['name'][:30]:30} -> {row['image_url']}")

print('\nChecking for missing image URLs:')
missing = df['image_url'].isna().sum()
print(f'Missing: {missing}/{len(df)} ({missing/len(df)*100:.1f}%)')

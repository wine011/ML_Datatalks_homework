import numpy as np
import pandas as pd


print(f"pandas version: {pd.__version__}")


# Read csv file
df = pd.read_csv('req_data_file.csv')
#print(df)

# Check the columns
print(f"Columns: {df.columns}")

print(df.info())

print(f"Unique value for 'ocean_proximity' column: {df.ocean_proximity.unique()}")

near_bay_df = df.loc[df['ocean_proximity'] == 'NEAR BAY']
print(f"Near_Bay_Dataframe:\n{near_bay_df}")

near_bay_median_house_value_df = near_bay_df.median_house_value
print(near_bay_median_house_value_df)
avg_value = near_bay_median_house_value_df.mean()
print(avg_value)


print(df.total_bedrooms)
avg_total_bedrooms_value = df.total_bedrooms.mean()
print(avg_total_bedrooms_value.round(3))

df["total_bedrooms"].fillna(avg_total_bedrooms_value.round(3), inplace=True)
print(df.info())

updated_avg_val = df.total_bedrooms.mean()
print(updated_avg_val.round(3))

# Located on islands
on_islands_df = df.loc[df['ocean_proximity'] == 'ISLAND']
print(f"On_islands_Dataframe: \n{on_islands_df}")

df1 = on_islands_df[["housing_median_age", "total_rooms", "total_bedrooms"]]
print(df1)

X = np.array(df1)
print(f"X:\n{X}")

XT = np.transpose(X)
print(f"XT:\n{XT}")

XTX = XT.dot(X)
print(f"XTX:\n{XTX}")

XTX_inv = np.linalg.inv(XTX)
print(f"XTX_inv: \n{XTX_inv}")

y = np.array([950, 1300, 800, 1000, 1300])
print(f"y:\n{y}")
re_y = y.reshape(5,1)
print(re_y.shape)
print(XT.shape)
z = XTX_inv.dot(XT)
print(f"z:\n{z}")
w = z.dot(re_y)
print(f"w:\n{w}")
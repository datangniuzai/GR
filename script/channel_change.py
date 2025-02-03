import pandas as pd
if __name__ == '__main__':
    new_order = [1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14, 41, 15, 16, 42, 17, 43, 18, 44, 19, 45, 
                20, 46, 21, 47, 22, 48, 23, 49, 24, 25, 50, 26, 51, 27, 52, 28, 53, 29, 54, 30, 55, 31, 
                56, 32, 33, 57, 34, 58, 35, 59, 36, 60, 37, 61, 38, 62, 39, 63, 40, 64]
    file_numbers = range(1, 27)
    for file_number in file_numbers:
        file_path = f'2025.1.14-26个手势-王泓雨/sEMG_data{file_number}.csv'
        df = pd.read_csv(file_path,header=None)
        original_columns = df.shape[1]
        if max(new_order) > original_columns:
            raise ValueError("指定的新顺序中存在超出原始列范围的索引！")
        new_columns = [i - 1 for i in new_order] 
        df_rearranged = df.iloc[:, new_columns]
        output_path = f'sEMG_data{file_number}.csv' 
        df_rearranged.to_csv(output_path, index=False,header =False)
        print(f"列已按照指定顺序重新排列，并保存到 {output_path}")

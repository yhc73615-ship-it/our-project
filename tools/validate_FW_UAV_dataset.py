import pandas as pd
import os

def validate_FW_UAV_dataset(data_path='dataset/FD'):
    """
    验证FW-UAV数据集是否符合DatasetFD类的数据运行需求
    """
    # 定义所需的文件
    required_files = ['train_X.csv', 'train_y.csv', 'test_X.csv', 'test_y.csv']
    
    print("开始验证FW-UAV数据集...")
    print(f"数据路径: {data_path}")
    print("="*50)
    
    # 检查所有必需文件是否存在
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少以下文件: {missing_files}")
        return False
    else:
        print("✅ 所有必需的CSV文件都存在")
    
    # 验证每个文件的格式
    validation_results = {}
    
    # 验证标签文件 (y files)
    y_files = ['train_y.csv', 'test_y.csv']
    for y_file in y_files:
        print(f"\n验证 {y_file}...")
        try:
            df = pd.read_csv(os.path.join(data_path, y_file))
            
            # 检查是否包含fault列
            if 'fault' not in df.columns:
                print(f"❌ {y_file} 缺少 'fault' 列")
                validation_results[y_file] = False
                continue
            
            # 检查fault列的值是否只包含0和1
            unique_values = set(df['fault'].unique())
            valid_values = {0, 1}
            if not unique_values.issubset(valid_values):
                print(f"❌ {y_file} 的 'fault' 列包含非0/1值: {unique_values - valid_values}")
                validation_results[y_file] = False
                continue
            
            print(f"✅ {y_file} 验证通过 - 包含'fault'列，值为0或1")
            validation_results[y_file] = True
            
        except Exception as e:
            print(f"❌ 读取 {y_file} 时出错: {str(e)}")
            validation_results[y_file] = False
    
    # 验证特征文件 (X files)
    X_files = ['train_X.csv', 'test_X.csv']
    for X_file in X_files:
        print(f"\n验证 {X_file}...")
        try:
            df = pd.read_csv(os.path.join(data_path, X_file))
            
            # 检查是否至少有一列特征
            if len(df.columns) == 0:
                print(f"❌ {X_file} 没有任何列")
                validation_results[X_file] = False
                continue
            
            # 检查是否至少有一些数据行
            if len(df) == 0:
                print(f"❌ {X_file} 没有任何数据行")
                validation_results[X_file] = False
                continue
            
            # 检查是否所有列都是数值型
            non_numeric_cols = []
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except:
                        non_numeric_cols.append(col)
            
            if non_numeric_cols:
                print(f"❌ {X_file} 包含非数值列: {non_numeric_cols}")
                validation_results[X_file] = False
                continue
            
            print(f"✅ {X_file} 验证通过 - 包含 {len(df.columns)} 个特征列，共 {len(df)} 行数据")
            validation_results[X_file] = True
            
        except Exception as e:
            print(f"❌ 读取 {X_file} 时出错: {str(e)}")
            validation_results[X_file] = False
    
    # 总结验证结果
    print("\n" + "="*50)
    print("验证结果总结:")
    all_passed = True
    for file, result in validation_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{file}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有验证通过！数据集符合DatasetFD类的数据运行需求。")
    else:
        print("\n⚠️  部分验证失败，请根据上述错误信息修正数据集。")
    
    return all_passed

if __name__ == "__main__":
    validate_FW_UAV_dataset()
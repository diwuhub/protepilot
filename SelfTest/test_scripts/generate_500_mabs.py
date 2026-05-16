import pandas as pd
import random

def generate_mock_mabs(num=500):
    # 真实的抗体骨架片段
    hc_fw1 = "EVQLVESGGGLVQPGGSLRLSCAASGFTFS"
    lc_fw1 = "DIQMTQSPSSLSASVGDRVTITC"
    
    data = []
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    for i in range(1, num + 1):
        # 随机生成 CDR 区模拟突变库
        hcdr = "".join(random.choices(amino_acids, k=random.randint(8, 15)))
        lcdr = "".join(random.choices(amino_acids, k=random.randint(6, 11)))
        
        hc = hc_fw1 + hcdr + "WVRQAPGKGLEWVAGIKQDGSEKYYVDSVKGRFTISRD"
        lc = lc_fw1 + lcdr + "WYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFT"
        
        data.append({
            "Candidate_ID": f"mAb_Candidate_{i:03d}",
            "Sequence_HC": hc,
            "Sequence_LC": lc
        })
        
    df = pd.DataFrame(data)
    df.to_csv("ht_screening_test_500.csv", index=False)
    print("✅ 成功生成 500 个抗体候选序列测试集: ht_screening_test_500.csv")

if __name__ == "__main__":
    generate_mock_mabs()
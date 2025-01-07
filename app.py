import streamlit as st
import pulp

def run_optimization(truck_data, demand_a95, demand_e5, demand_diesel):
    """
    Triển khai phương pháp hai bước (two-phase):
      1) Giảm thiểu số xe tải (trucks) được sử dụng.
      2) Trong các phương án sử dụng ít xe tải nhất, giảm thiểu lượng dư thừa (leftover).

    Tham số:
      - truck_data: dict {tên_xe: [(mã_ngăn, sức_chứa), ...]}
      - demand_a95, demand_e5, demand_diesel: nhu cầu sử dụng từng loại
    """

    total_demand = demand_a95 + demand_e5 + demand_diesel

    # ---------------- BƯỚC 1: Giảm thiểu số xe tải -------------------
    phase1 = pulp.LpProblem("GiamThieuSoXeTai", pulp.LpMinimize)

    # X[t] = 1 nếu xe tải t được sử dụng
    X = {}
    # Biến nhị phân cho từng ngăn: A_c, E_c, D_c
    A = {}
    E = {}
    D = {}

    # Tạo danh sách tất cả các ngăn
    all_compartments = []
    for t, compartments in truck_data.items():
        for (cid, cap) in compartments:
            all_compartments.append((t, cid, cap))

    for t in truck_data:
        X[t] = pulp.LpVariable(f"UseTruck_{t}", cat=pulp.LpBinary)

    for (t, cid, cap) in all_compartments:
        A[(t, cid)] = pulp.LpVariable(f"A95_{t}_{cid}", cat=pulp.LpBinary)
        E[(t, cid)] = pulp.LpVariable(f"E5_{t}_{cid}", cat=pulp.LpBinary)
        D[(t, cid)] = pulp.LpVariable(f"D_{t}_{cid}", cat=pulp.LpBinary)

    # Mỗi ngăn chỉ dùng cho tối đa một loại xăng/dầu
    for (t, cid, cap) in all_compartments:
        phase1 += A[(t, cid)] + E[(t, cid)] + D[(t, cid)] <= 1
        # Nếu ngăn này được dùng, xe tải phải được sử dụng
        phase1 += A[(t, cid)] + E[(t, cid)] + D[(t, cid)] <= X[t]

    # Đáp ứng nhu cầu
    phase1 += pulp.lpSum(A[(t, cid)] * cap for (t, cid, cap) in all_compartments) >= demand_a95
    phase1 += pulp.lpSum(E[(t, cid)] * cap for (t, cid, cap) in all_compartments) >= demand_e5
    phase1 += pulp.lpSum(D[(t, cid)] * cap for (t, cid, cap) in all_compartments) >= demand_diesel

    # Mục tiêu Bước 1: Minimize tổng số xe tải
    phase1.setObjective(pulp.lpSum(X[t] for t in truck_data))

    # Giải Bước 1
    phase1.solve(pulp.PULP_CBC_CMD(msg=0))
    if phase1.status != pulp.LpStatusOptimal:
        return None  # không tìm thấy lời giải

    min_trucks = sum(pulp.value(X[t]) for t in truck_data)

    # --------------- BƯỚC 2: Cố định số xe tải, giảm dư thừa ----------
    phase2 = pulp.LpProblem("GiamDuThua", pulp.LpMinimize)

    X2 = {}
    A2 = {}
    E2 = {}
    D2 = {}

    for t in truck_data:
        X2[t] = pulp.LpVariable(f"UseTruck2_{t}", cat=pulp.LpBinary)

    for (t, cid, cap) in all_compartments:
        A2[(t, cid)] = pulp.LpVariable(f"A95_2_{t}_{cid}", cat=pulp.LpBinary)
        E2[(t, cid)] = pulp.LpVariable(f"E5_2_{t}_{cid}", cat=pulp.LpBinary)
        D2[(t, cid)] = pulp.LpVariable(f"D_2_{t}_{cid}", cat=pulp.LpBinary)

    for (t, cid, cap) in all_compartments:
        phase2 += A2[(t, cid)] + E2[(t, cid)] + D2[(t, cid)] <= 1
        phase2 += A2[(t, cid)] + E2[(t, cid)] + D2[(t, cid)] <= X2[t]

    phase2 += pulp.lpSum(A2[(t, cid)] * cap for (t, cid, cap) in all_compartments) >= demand_a95
    phase2 += pulp.lpSum(E2[(t, cid)] * cap for (t, cid, cap) in all_compartments) >= demand_e5
    phase2 += pulp.lpSum(D2[(t, cid)] * cap for (t, cid, cap) in all_compartments) >= demand_diesel

    # Phải dùng đúng số xe tải = min_trucks
    phase2 += pulp.lpSum(X2[t] for t in truck_data) == min_trucks

    # Mục tiêu Bước 2: Giảm thiểu tổng dung tích sử dụng (tương đương giảm dư thừa)
    phase2.setObjective(
        pulp.lpSum((A2[(t, cid)] + E2[(t, cid)] + D2[(t, cid)]) * cap for (t, cid, cap) in all_compartments)
    )

    phase2.solve(pulp.PULP_CBC_CMD(msg=0))
    if phase2.status != pulp.LpStatusOptimal:
        return None

    total_used = pulp.value(
        pulp.lpSum((A2[(t, cid)] + E2[(t, cid)] + D2[(t, cid)]) * cap for (t, cid, cap) in all_compartments)
    )
    leftover = total_used - total_demand

    # Lấy kết quả
    used_trucks = []
    allocation = []

    for t in truck_data:
        if pulp.value(X2[t]) > 0.5:
            used_trucks.append(t)

    for (t, cid, cap) in all_compartments:
        valA = pulp.value(A2[(t, cid)])
        valE = pulp.value(E2[(t, cid)])
        valD = pulp.value(D2[(t, cid)])
        if (valA + valE + valD) > 0.5:
            if valA > 0.5:
                product = "A95"
            elif valE > 0.5:
                product = "E5"
            else:
                product = "Dầu"
            allocation.append((t, cid, cap, product))

    return {
        "min_trucks": int(min_trucks),
        "total_used": total_used,
        "leftover": leftover,
        "used_trucks": used_trucks,
        "allocation": allocation,
    }


def main():
    st.title("Ứng Dụng Tối Ưu Xe Tải Cho A95, E5, Dầu")

    st.markdown("""
    **Hướng dẫn**:
    1. Nhập nhu cầu cho A95, E5, Dầu trong thanh bên (sidebar).
    2. Thêm danh sách xe tải và các ngăn.
    3. Bấm nút "Tính Toán" để xem:
       - Số xe tải **tối thiểu** cần dùng.
       - Phân bổ các ngăn sao cho lượng **dư thừa** là nhỏ nhất.
    """)

    # Khu vực sidebar để nhập nhu cầu
    st.sidebar.header("Nhu Cầu Sử Dụng")
    demand_a95 = st.sidebar.number_input("Nhu cầu A95", value=26600, min_value=0)
    demand_e5 = st.sidebar.number_input("Nhu cầu E5", value=9800, min_value=0)
    demand_diesel = st.sidebar.number_input("Nhu cầu Dầu", value=12600, min_value=0)

    st.sidebar.header("Xe Tải và Ngăn Chứa")
    st.sidebar.markdown("Nhập danh sách xe tải theo định dạng: `TenXe: c1,c2,c3,...`")
    st.sidebar.markdown("Ví dụ: `T1_43C_22820: 5000,5000,4000,4000,2000`")

    user_input = st.sidebar.text_area(
        "Danh sách xe tải",
        value=(
            "T1_43C_22820: 5000,5000,4000,4000,2000\n"
            "T2_43H_04420: 6000,5000,4000,3000,2000\n"
            "T3_43C_09424: 3080,4220,4550,4600\n"
            "T4_43C_08079: 3150,4150,4250,4650,4700\n"
            "T5_43C_09410: 2000,4000,5000,5000,5000\n"
            "T6_43C_09666: 3160,4160,4600,4600\n"
            "T7_43S_07908: 6160,5160,4100,3200\n"
        )
    )

    if st.sidebar.button("Tính Toán"):
        # Xử lý chuỗi input -> truck_data
        truck_data = {}
        lines = user_input.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                st.error(f"Dòng không hợp lệ (thiếu dấu hai chấm): {line}")
                return
            truck_name, comps_str = line.split(":", 1)
            truck_name = truck_name.strip()
            comps_list = comps_str.strip().split(",")
            comp_array = []
            idx = 1
            for cstr in comps_list:
                cstr = cstr.strip()
                if not cstr.isdigit():
                    st.error(f"Dung tích không hợp lệ '{cstr}' ở dòng: {line}")
                    return
                cap_val = int(cstr)
                comp_id = f"{truck_name}_c{idx}"
                idx += 1
                comp_array.append((comp_id, cap_val))
            truck_data[truck_name] = comp_array

        result = run_optimization(truck_data, demand_a95, demand_e5, demand_diesel)
        if not result:
            st.error("Không tìm thấy lời giải khả thi. Vui lòng kiểm tra dữ liệu hoặc nhu cầu.")
            return

        st.success("Đã tìm thấy phương án tối ưu!")
        st.write(f"**Số xe tải tối thiểu cần dùng** = {result['min_trucks']}")
        st.write(f"**Lượng dư thừa** = {int(result['leftover'])} L "
                 f"(Tổng đã dùng = {int(result['total_used'])} L)")

        st.write("**Các xe tải thực sự sử dụng**:")
        st.write(result["used_trucks"])

        st.write("**Chi tiết phân bổ ngăn**:")
        df_alloc = []
        for (truck, cid, cap, product) in result["allocation"]:
            df_alloc.append({
                "Xe Tải": truck,
                "Ngăn": cid,
                "Sức Chứa": cap,
                "Loại": product
            })
        st.table(df_alloc)


if __name__ == "__main__":
    main()



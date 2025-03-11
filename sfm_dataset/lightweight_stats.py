import json
from collections import defaultdict


# 2024/05/30 07:20:30               456  000/000/colmap/0/cameras.bin
# 2024/05/30 07:20:32           1902198  000/000/colmap/0/images.bin
# 2024/05/30 07:20:31            200555  000/000/colmap/0/points3D.bin

def get_key_from_short_index(short_i):
    short_i = str(short_i)
    i = "0" * (6 - len(short_i)) + short_i
    i = i[:3] + "/" + i[3:]
    return i


def main():

    with open("../data/sfm_work/categories.json", "rb") as fd:
        js = json.load(fd)

    key_names = {get_key_from_short_index(short_i): name for name, short_i in js.items()}

    with open('../data/ls_reconstructs.txt', 'r') as f:
        lines = f.readlines()

    m = defaultdict(lambda: defaultdict(int))

    for line in lines:
        tokens = [t.strip() for t in line.split(' ') if len(t) > 0]
        s = int(tokens[2])
        key_all = tokens[3]
        key_t = key_all.split("/")[:2] + key_all.split("/")[4:-1]
        key_t = "/".join(key_t)

        key_data = key_all.split("/")[-1]
        m[key_data][key_t] += s

        #print(f"{s=}, {key_t=}, {key_all=}, {key_data=}")

    for data_k, v in m.items():
        ks_sums = sorted(v.items(), key=lambda x: x[1], reverse=True)
        not_present = [k for k, _ in ks_sums if k not in key_names]
        ns_ks_sums = [(k, key_names[k], s) for k, s in ks_sums if k in key_names]

        first = 100
        print()
        print(f"first {first} for: {data_k}")
        def str_me(s):
            only_keys = False
            if only_keys:
                return f"'{s[0]}' '{s[1]}'"
            else:
                return f"'{s[0]}' '{s[1]}'   {s[2]}"

        print("\n".join([str_me(s) for s in ns_ks_sums[:first]]))

        print("not present")
        print("\n".join(not_present))


if __name__ == "__main__":
    main()

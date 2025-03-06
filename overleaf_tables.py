
def print_bulk(model_data_rows, model_data_rows_s, model_lines, model_data_indices):
    assert len(model_data_rows) == 3
    assert len(model_data_rows_s) == 3

    def smart_strip_list(l):
        r = []
        for o, i in enumerate(l):
            if i.startswith(" ") and o > 0:
                i = i[1:]
            if i.endswith(" "):
                i = i[:-1]
            r.append(i)
        return r

    row_s = [smart_strip_list(model_data_rows_s[i][:2]) for i in range(3)]

    for col in range(Conf.cols):
        values = [model_data_rows[i][col] for i in range(3)]
        values = sorted(values, reverse=Conf.cols_higher_better[col])
        for i in range(3):
            slash = "\\"
            if model_data_rows[i][col] == values[0]:
                row_s[i].append(f"\\textBF{{{model_data_rows_s[i][2 + col].replace(slash, ' ').strip()}}}")
            elif model_data_rows[i][col] == values[1]:
                row_s[i].append(f"\\underline{{{model_data_rows_s[i][2 + col].replace(slash, ' ').strip()}}}")
            else:
                assert model_data_rows[i][col] == values[2]
                row_s[i].append(model_data_rows_s[i][2 + col].replace(slash, ' ').strip())

    assert len(model_data_indices) == 3
    for order, i in enumerate(model_data_indices):
        model_lines[i] = " & ".join(row_s[order]) + " \\\\"

    for ml in model_lines:
        print(ml)


def main(infp):

    start = "start"
    end_s = "end"
    # states = [end_s, start]
    state = end_s

    for line in open(infp).readlines():
        # line = line.strip()
        if line.strip().startswith("%"):
            assert state == start
            model_lines.append(line[:-1]) # no line feed
            model_data_index += 1
            continue
        elif line.strip().startswith("\\midrule") or line.strip().startswith("\\bottomrule"):
            assert state == start
            model_lines.append(line[:-1]) # no line feed
            model_data_index += 1
            state = end_s

            print_bulk(model_data_rows, model_data_rows_s, model_lines, model_data_indices)
            continue

        elif line.strip().startswith("\\multirow"):
            assert state == end_s
            state = start
            model_lines = []
            model_data_rows = []
            model_data_rows_s = []
            model_data_indices = []
            model_data_index = 0

        tokens = line.split("&")
        line_numbers = [float(i.replace("\\", " ")) for i in tokens[2:]]
        line_s_n = [i for i in tokens]
        assert len(line_numbers) == Conf.cols
        model_data_rows.append(line_numbers)
        model_data_rows_s.append(line_s_n)
        model_lines.append(line)
        model_data_indices.append(model_data_index)
        model_data_index += 1


    if state == start:
        print_bulk(model_data_rows, model_data_rows_s, model_lines, model_data_indices)


class Conf:
    # cols = 10
    # cols_higher_better = [False, False, True, True, False, False, False, True, True, False]

    cols = 14
    cols_higher_better = [False, False, False, True, True, True, False, False, False, False, True, True, True, False]


if __name__ == "__main__":
    #main(infp="exps_overleaf/bu_in_1.txt")
    main(infp="exps_overleaf/bu_in_2.txt")
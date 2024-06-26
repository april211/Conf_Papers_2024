



if __name__ == '__main__':

    # write the full version
    paper_cnt = -1
    with open('icml2024_full.md', 'w') as res_file:
        res_file.write('# ICML 2024\n\n')
        with open('res/raw/icml/icml_ms.md', 'r') as src_file:
            title_flag = True
            for line in src_file:
                if title_flag:                  # title line
                    paper_cnt += 1
                    res_file.write('## ' + f"{paper_cnt}" + ". " + line)
                    title_flag = False
                elif len(line) <= 5:            # empty line
                    title_flag = True
                    res_file.write('\n')
                else:                           # abstract line
                    res_file.write(line)

    # write the folded version
    paper_cnt = -1
    with open('icml2024_folded.md', 'w') as res_file:
        res_file.write('# ICML 2024\n\n')
        with open('res/raw/icml/icml_ms.md', 'r') as src_file:
            title_flag = True
            for line in src_file:
                if title_flag:                  # title line
                    paper_cnt += 1
                    res_file.write('## ' + f"{paper_cnt}" + ". " + line)
                    title_flag = False
                elif len(line) <= 5:            # empty line
                    title_flag = True
                    res_file.write('\n')
                else:                           # abstract line
                    res_file.write("\n<details>\n\n")
                    res_file.write("<summary>Abstract</summary>\n\n")
                    res_file.write(line)
                    res_file.write("\n</details>\n")


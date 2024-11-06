# Description: This script is used to generate the markdown files of the conference paper list.



# the header of the markdown file
header = '# NeurIPS 2024\n\n'      # '# ICML 2024\n\n', '# IJCAI 2024\n\n', '# KDD 2024\n\n', '# NeurIPS 2024\n\n'

# the conference name
conf_name = 'neurips'              # icml, ijcai, kdd, neurips




if __name__ == '__main__':

    # write the full version
    paper_cnt = -1
    with open(f'{conf_name}2024_full.md', 'w') as res_file:
        res_file.write(header)
        with open(f'res/raw/{conf_name}/{conf_name}_ms.md', 'r') as src_file:
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
    with open(f'{conf_name}2024_folded.md', 'w') as res_file:
        res_file.write(header)
        with open(f'res/raw/{conf_name}/{conf_name}_ms.md', 'r') as src_file:
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


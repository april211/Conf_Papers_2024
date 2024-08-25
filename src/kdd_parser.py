import bibtexparser
import pandas as pd


# read the bib file
# library = bibtexparser.parse_file("data/KDD_2024.bib")
library = bibtexparser.parse_file("data/KDD_2024_corrected.bib")

if len(library.failed_blocks) > 0:
    print("Some blocks failed to parse. Check the entries of `library.failed_blocks`.")
else:
    print("All blocks parsed successfully")

print(f"Parsed {len(library.blocks)} blocks, including:"
  f"\n\t{len(library.entries)} entries"
    f"\n\t{len(library.comments)} comments"
    f"\n\t{len(library.strings)} strings and"
    f"\n\t{len(library.preambles)} preambles")

# correct the wrong entries
# print(library.comments[0].comment)

print("Failed blocks:", len(library.failed_blocks))

for failed_block in library.failed_blocks:
    print(failed_block.start_line)
    print(failed_block.raw)
    print()

# extract the `title` & `abstract` fields
paper_list = []
for entry in library.entries:
    paper_list.append({
        "paper_title": entry["title"],
        "abstract": entry["abstract"]
    })

# convert to pandas dataframe
paper_df = pd.DataFrame(paper_list)

# save to the file
paper_df.to_csv("data/KDD_2024.csv", index=False)


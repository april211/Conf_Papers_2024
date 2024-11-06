# Conference Papers 2024

Interesting papers related to graph data mining.

You can also use the code provided in this repository to **generate your own conference papers list**.

## List of Conferences

1. ICML 2024: [folded_version](./icml2024_folded.md), [full_version](./icml2024_full.md), [PDFs](https://openreview.net/group?id=ICML.cc/2024/Conference)

2. IJCAI 2024: [folded_version](./ijcai2024_folded.md), [full_version](./ijcai2024_full.md), [PDFs](https://www.ijcai.org/Proceedings/2024/)

3. KDD 2024: [folded_version](./kdd2024_folded.md), [full_version](./kdd2024_full.md), [PDFs](https://dl.acm.org/doi/proceedings/10.1145/3637528)

4. NeurIPS 2024: [folded_version](./neurips2024_folded.md), [full_version](./neurips2024_full.md), [PDFs](https://openreview.net/group?id=NeurIPS.cc/2024/Conference#tab-accept-oral)

## How to Generate Your Own Conference Paper List?

The workflow of generating conference paper list is as follows:

1. Get the raw conference papers data from the conference website or other sources, and put it into the `data` folder.

2. Process the raw data using the provided scripts in the `src` folder. You can also write your own scripts to process the data. Make sure you convert the raw data into the `csv` format and put it into the `data` folder.

3. Run the `xxx_filter` script to filter the papers you are interested in. The filtered papers will be saved in the `res` folder (`xxx_results.md`).

4. Generally, you need to double-check the filtered papers and manually adjust the content if necessary (`xxx_ms.md`).

5. Run the `markdown_formatter.py` script to generate the final markdown file (`xxx_folded.md` and `xxx_full.md`).

Currently, the filtering mechanism is simply **based on keywords**. You can adjust the keywords in the `data` folder.


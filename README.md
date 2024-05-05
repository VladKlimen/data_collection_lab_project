# data_collection_lab_project
The project focuses on scrapping the job positions posts from Linkedin, clustering the data like skills and titles, and visualizing the clusters in a pretty and interactive way. See the notebooks for more info and code using.

Files and folders description:
1. Models - contains two clustering data pickles - for job titles and for skills. In clusters_vis.ipynb you can find a code for visualization that uses these pickles. Also, there are two demo htmls - the interactive clusters plots.
2. part_5_packages - py files of functions used in the analysis part and clustering model creation - used in 5_jobs_data_analysis.ipynb.
3. 1_data_preparation_for_scraping.ipynb - extracting and preparing the relevant data from companies dataset (provided in the project).
4. Project - scraping.ipynb - the notebook (well documented) for jobs scraping. Using Bright Data datacenter proxies.
5. 2_research_question_1.ipynb - answering the research question from our project proposal (How likely for a company from each industry and size to be a hiring company and what is
the distribution of jobs number?)
6. 3_gemini_jobs_parsing.ipynb - jobs data parsing using Google's Gemini LLM.
7. 4_jobs_data_fixing.ipynb - exploring and fixing some errors of the Gemini's responses.
8. 5_jobs_data_analysis.ipynb - the main notebook that contains all the model code and jobs data processing (preprocessing, embedding, clustering, exploring the clusters)
9. trash.ipynb - some trash code that was moved from other notebooks (useless for now).

installed = installed.packages()[,'Package']
needed = c("tibble","dplyr","tidyr","purrr","readr", 
           "parallel","doParallel",
          "mda","pls","sparseLDA","MASS","caret",
          "Matrix","spls","e1071","elasticnet",
          "pbmcapply",'extraaa')

install.packages(setdiff(needed, installed))
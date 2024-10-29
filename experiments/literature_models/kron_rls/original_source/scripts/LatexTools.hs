module LatexTools where

import Data.List

--------------------------------------------------------------------------------
-- Tools for latex tables
--------------------------------------------------------------------------------

type Table = [[String]]

-- | Add a horizontal (multirow) heading
addHead :: String -> Table -> Table
addHead y [xs]     = [y:xs]
addHead y (xs:xss) = (header:xs):(map ("":) xss)
  where len    = length xss+1
        header = "\\multirow{" ++ show len ++ "}{*}{" ++ y ++ "}\n"

hline :: Table
hline = [[]]

vconcat :: [Table] -> Table
vconcat = concat

showTable :: Table -> String
showTable xss = unlines $
    ["\\begin{tabular}{" ++ replicate width 'l' ++ "}"]
 ++ (map showRow xss ++ [])
 ++ ["\\end{tabular}"]
  where width = maximum $ map length xss
        showRow [] = "\\hline"
        showRow xs = intercalate " & " xs ++ "\\\\"


module Main where

import LatexTools
import Bag
import System.Environment
import Data.List
import qualified Data.Map as Map
import Text.Printf

toTable :: Bool -> [Bag String] -> Table
toTable full items
  = hline
 ++ (["Dataset","Method","Kernel"] ++ map snd fields)
  : concat
   [ (hline++) . addHead dn . concat $
   --  [ addHead kn . concat $
       [ [ mn ++ [ formatCell $ item Map.! f | (f,_) <- fields ]]
       | (m,mn) <- methods
       , let item = onlyOne (m ++ "\n from:\n" ++ show items3) $ filter (matchItem m) items3
       ]
   --  | (k,kn) <- kernels
   --  , let items3 = filter (matchItem k) items
   --  ]
   | (d,dn) <- datasets
   , let items2 = atLeastOne (d ++ "\n from:\n" ++ show items) $ filter (matchItem d) items
   , let items3 = items2
   ] ++ hline
 where
  fields
    = [("auc","AUC")]
   ++ [("aupr","AUPR")]
   ++ [("sensitivity_1_100","Sensitivity") | full]
   ++ [("specificity_1_100","Specificity") | full]
   ++ [("ppv_1_100","PPV") | full]
  methods
    = --[("fun:predict_laprls","NetLapRLS")
      [("fun:*predict_rls, 0, 0*",["RLS-avg","chem/gen"])
      ,("fun:*predict_rls, 1, 1*",["RLS-avg","GIP"])
      ,("fun:*predict_rls, 0.5, 0.5*",["RLS-avg","avg."])
      ,("fun:*predict_rls_kron, 0, 0*",["RLS-Kron","chem/gen"])
      ,("fun:*predict_rls_kron, 1, 1*",["RLS-Kron","GIP"])
      ,("fun:*predict_rls_kron, 0.5, 0.5*",["RLS-Kron","avg."])]
  kernels
    = --[("fun:predict_laprls","NetLapRLS")
      [("fun:*, 0, 0*","chem/seq")
      ,("fun:*, 1, 1*","IPK")
      ,("fun:*, 0.5, 0.5*","avg.")]
  datasets
    = [("data:drugtarget-e*","Enzyme")
      ,("data:drugtarget-ic*","Ion Channel")
      ,("data:drugtarget-gpcr*","GPCR")
      ,("data:drugtarget-nr*","Nuclear Receptor")]
  --kernels = [("*-nokernel","None"),("*-chem/seq","Chem/Seq")]

formatCell :: String -> String
--formatCell = id--takeWhile (\x -> x `elem` ('.':['0'..'9']))
-- Format a number as %, with std deviation in parentheses
formatCell xs = case span (/= ' ') xs of
 (as,' ':'(':ys)
   | a >= 100  -> printf ("\\meanstddev{%.f}{"++bprec++"}") a b
   | a >= 99.5 -> printf ("\\meanstddev{%.2f}{"++bprec++"}") a b
   | otherwise -> printf ("\\meanstddev{%.1f}{"++bprec++"}") a b
     where (bs,")")        = span (/= ')') ys
           a = read as * 100 :: Double
           b = read bs * 100 :: Double
           bprec | b < 0.01  = "%.3f"
                 | b < 0.1   = "%.2f"
                 | otherwise = "%.1f"
 _
  -- | a >= 100  -> printf ("\\meanonly{%.f}") a
  -- | a >= 99.5 -> printf ("\\meanonly{%.2f}") a
   | otherwise -> printf ("\\meanonly{%.1f}") a
     where a = read xs * 100 :: Double

matchItem :: String -> Bag String -> Bool
matchItem query item = matchString b (item Map.! a)
  where (a,_:b) = break (==':') query

matchString :: String -> String -> Bool
matchString ('*':xs) ys     = any (matchString xs) (tails ys)
matchString (x:xs)   (y:ys) = x == y && matchString xs ys
matchString []       []     = True
matchString _        _      = False

onlyOne msg [x] = x
onlyOne msg []  = error $ "No items match the query: " ++ msg
onlyOne msg xs  = error $ "Multiple items match the query: "++msg++"\n" ++ show xs

atLeastOne msg [] = error $ "No items match the query: " ++ msg
atLeastOne msg xs = xs

filterPartition :: [a -> Bool] -> [a] -> [[a]]
filterPartition = undefined

main = do
    args <- getArgs
    let (full,queries) = case args of
                           ("--full":xs) -> (True,xs)
                           xs -> (False,xs)
    body <- getContents
    if null body
     then do
       putStrLn "Usage: runhaskell MakeTexTable [--full] 'field:value' < results from run_all_tests"
     else do
       let page = showTable
                . toTable full
                . atLeastOne (concat $ intersperse " " queries)
                . filter (\i -> all (\q -> matchItem q i) queries)
                . parse $ body
       putStrLn $ page

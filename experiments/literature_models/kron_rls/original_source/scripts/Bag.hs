{-# LANGUAGE
         TypeSynonymInstances
  #-}

module Bag where

import Control.Monad.State
import Control.Applicative hiding (many)
import Data.Char
import Text.Parsec hiding (parse,(<|>),State)
import qualified Text.Parsec as P
import Data.Map (Map)
import qualified Data.Map as Map

--------------------------------------------------------------------------------
-- Util
--------------------------------------------------------------------------------

trim, ltrim :: String -> String
trim = ltrim . reverse . ltrim . reverse
ltrim = dropWhile isSpace

--------------------------------------------------------------------------------
-- Type
--------------------------------------------------------------------------------

type Bag a = Map String a

--------------------------------------------------------------------------------
-- Parsing
--------------------------------------------------------------------------------

type Parser = Parsec String ()

--sp = many (oneOf " \t")
--word = (:) <$> satisfy isWordStart <:> many1 (satisfy isWordRest)
--  where isWordStart x = isAlpha x || x `elem` 

parserStruct, parserStructBody :: Parser a -> Parser (Bag a)
parserStruct pValue = spaces *> oneOf "{" *> parserStructBody pValue <* oneOf "}" <* spaces

parserStructBody pValue = Map.fromList <$> (spaces *> many line <* spaces)
  where line = do
                  spaces
                  key <- many (noneOf "\n:=)}]") <?> "key"
                  oneOf ":="
                  many (oneOf " \t") <?> ""
                  value <- pValue
                  many (oneOf "\n,") <?> ""
                  spaces
                  return (trim key,value)
             <?> "key/value pair"

-- | Parse a value with nested parentheses
parserValue :: Parser String
parserValue = trim <$> body <?> "value"
    where body = concat <$> many (anyParens <|> return <$> noneOf "\n,)}]")
          anyParens = parens '(' ')' <|> parens '{' '}' <|> parens '[' ']'
          parens o c = (\a b c -> [a]++b++[c]) <$> char o <*> parenBody c <*> char c
          parenBody c = concat <$> many (anyParens <|> return <$> noneOf [c])

parser :: Parser [Bag String]
parser = spaces *> many (parserStruct parserValue) <* spaces <* eof

parse :: String -> [Bag String]
parse xs = case P.parse parser "" xs of
          Left  e -> error (show e)
          Right x -> x


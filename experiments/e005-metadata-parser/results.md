# E005 result — partial support, targeted repair required

The initial conservative parser extracted at least one year from 1,011 of 1,012
questions and a high-confidence separate-scope cue from 365 questions.

Entity coverage was only 605/1,012. Inspection showed that the dominant cause
was not ambiguous company naming: public questions frequently use a bare ticker
such as `SCR` or `VSC`, while the initial rule only recognized tickers in
parentheses. This result is retained as a failed baseline and motivates E005b.


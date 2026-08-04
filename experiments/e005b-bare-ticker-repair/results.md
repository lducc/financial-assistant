# E005b result — supported

Adding only exact bare tokens found in the supplied 100-ticker map raises entity
coverage from 605 to 983 of 1,012 questions (+378; 97.1% coverage). Twenty-nine
questions remain unresolved. The parser emits an audit source for every entity:
`explicit_bare_ticker`, `explicit_parenthetical_ticker`, or `company_alias`.

This is coverage, not gold entity-linking accuracy. The filter must be checked
against the frozen E003 proxy annotations before it is promoted from a safe
candidate generator to a retrieval hard filter.


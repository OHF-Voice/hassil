# Template Syntax

hassil's template syntax compactly represents large numbers of sentences for the purpose of intent recognition. It borrows inspiration from [Rhasspy's template language][rhasspy], which itself borrows from [JSGF][].

## Reserved Characters

The following characters are reserved for special syntax: `()[]{}<>|@`
These characters can be escaped with a backslash: `\`

Everything else (except whitespace) is matched literally, so it is not advised to have punctuation in your sentence templates.

## Optionals

Template block that may be omitted, surrounded by `[]`
Example: `[the]`

An optional `[x]` is equivalent to the alternative `(x|)`

## Alternatives

Example: `(red|green|blue)`

Template blocks separated by `|` and surrounded by `()` or `[]`
Blocks surrounded by `()` are required while `[]` is optional.

## Permutations

Example: `(TV; pause)`

Template blocks separated by `;` and surrounded by `()`
All possible orderings of the blocks are considered.

## Lists

Example: `{list_name}`

Reference to a pre-defined list. The matched list value will be copied to an entity with the same name as the list.

### Slot Name

Example: `{list_name:slot_name}`

Copies the matched list value to a differently named slot.

### Captures

Example: `{list_name:@capture_name}`

Copies the matched list text to a "capture", which is intended for use in the response (not a slot).

Example `{@list_name}`

Compact syntax equivalent to `{list_name:@list_name}`

## Expansion Rules

Example: `<rule_name>`

Reference to a named template block that is substituted in place of the reference.
Rules cannot reference themselves.

<!-- Links -->
[JSGF]: https://www.w3.org/TR/jsgf/
[rhasspy]: https://rhasspy.readthedocs.io/en/latest/training/

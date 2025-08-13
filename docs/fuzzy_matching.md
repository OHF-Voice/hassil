# Fuzzy Matching

Matching in hassil is strict. The template `set lights to (red|green|blue)` cannot match the sentence "red lights", for example.
Fuzzy matching allows hassil to recognize sentences outside of this strict set using [existing sentence templates][intents] as training material.

Our approach to fuzzy matching is inspired by [Mycroft's Adapt][adapt] but with several important differences:

* Hassil sentence templates are training material
* Automatically discovered groups of words (n-grams) instead of hand-coded keywords and phrases
* Multiple interpretations of input sentence constrained by possible [slot combinations][]
* Common words like "ok" and "please" are ignored
* Unique intent words can be used to break ties

## Training

With **fuzzy matching**, hassil can take advantage of [n-gram][] models to match sentences beyond the templates alone. An n-gram model contains the probabilities of word groups with the size of the largest group called the model's **order**. For example, an order 3 n-gram model of the template above would have probabilities for groups like "set", "set lights", "set lights to", "lights to red", and so on.

Because all possible word groups will never be present in the training data, n-gram models use different "smoothing" methods to guess their probabilities. hassil's fuzzy matching models are order 4 with [Kneser-Ney smoothing][kneser-ney]. Different orders were tested, with 3 being too low and 5 not providing much more benefit.

An n-gram model is trained per intent, and in some cases per domain and intent. The `HassTurnOn` intent is used to turn on lights as well as open covers. To avoid the sentence "open the office lights" turning them on, a separate `cover_HassTurnOn` intent is trained. The same is done for locks, etc.

During training, list names are kept as placeholder "words" like in `set {name} to {color}`. This allows lists like `{name}` and `{area}` to be filled in later with the user's entity names and areas. This helps keep the n-gram models small since number lists like `{brightness}` are not expanded to all possible values.

## Algorithm

When matching a sentence, different interpretations of the sentence are generated and checked. Before matching, a [trie][] is filled with possible list values like entity names (`{name}`), numbers (`{brightness}`), and language-specific values (`{color}`). Sequences of words (called "spans") are annotated with every list they could match. Some spans could have many possibilities, like "50" being a `{brightness}`, `{position}`, or `{temperature}`.

If the sentence "set office lights to 50" is to be matched, some obvious interpretations are:

* `set {area} {domain} to {brightness}`
    * `{area}` is "office"
    * `{domain}` is `light`
    * `{brightness}` is 50
* `set {name} to {brightness}`
    * `{name}` is "office lights"
    * `{brightness}` is 50
    
Assuming we have a light entity named "office lights", there are more interpretations, such as:

* `set {area} lights to {temperature}`
* `set {name} to {position}`

and even nonsensical ones like:

* `set {area} {domain} to {position}`

Before using the probabilities from each intent's n-gram model, we need to filter out interpretations that are obviously wrong. This is done with the help of [slot combinations][] and by tracking the entity domain coming either from `{name}` or `{domain}`.

With slot combinations alone, we can eliminate `set {area} {domain} to {position}` because `HassSetPosition` is the only intent with `{position}` and there is no valid slot combination for `{area}`, `{domain}`, and `{position}`. Using the fact that "office lights" is a light entity, we can also eliminate `set {name} to {position}` because `HassSetPosition` is only valid for cover and valve entities.

We will need to check the n-gram models for the remaining interpretations:

* `set {area} {domain} to {brightness}`
* `set {name} to {brightness}`
* `set {area} lights to {temperature}`

The word "lights" in `set {area} lights to {temperature}` will not be present in the vocabulary for the `HassClimateSetTemperature` intent's n-gram model. This means it will be assigned an "unknown" or "out-of-vocabulary" probability as a penalty. In addition to this penalty, hassil checks how likely the word is in the *other* intent n-gram models. The more likely, the worse the penalty will be with the reasoning that these words are important for distinguishing between intents. With this penalty, we should expect the interpretation to be ranked lowest. 

The final two interpretations will have similar probabilities from `HassLightSet`. To break the tie, hassil will boost interpretations with `{name}` if the probabilities are close enough. This rule is elsewhere in hassil and Home Assistant: prefer entity names where possible. So the final interpretation will be: `set {name} to {brightness}`.

To avoid ambiguity, the sentence `{name}` is always considered unmatched. Additionally, if interpretations from two different intents are scored very closely, the sentence is considered unmatched. This is an attempt to avoid cases where minor probability differences in the n-gram models would cause an entity to be turned on or off due to (seemingly) unrelated words. Without these protections, sentences like "garage door", "the garage door", and "garage door the" could do different things.

Lastly, "skip" and [stop words][] are used to ignore things like "could you please" and "ok". Every n-gram model is trained with special `<skip>` "words" at the beginning and end of every sentence. During matching, phrases like "could you please" are replaced with `<skip>`. Multiple `<skip>` words are combined, so "could you please please please" will only produce a single one. Stop words, like "that" or "this", may occur within a sentence and are not given the full penalty that other unknown/out-of-vocabulary words are.

<!-- Links -->
[n-gram]: https://en.wikipedia.org/wiki/Word_n-gram_language_model
[kneser-ney]: https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing
[trie]: https://en.wikipedia.org/wiki/Trie
[slot combinations]: https://github.com/OHF-Voice/intents/blob/main/docs/slot_combinations.md
[stop words]: https://en.wikipedia.org/wiki/Stop_word
[intents]: https://github.com/OHF-Voice/intents
[adapt]: https://github.com/MycroftAI/adapt
[slot combinations]: https://github.com/OHF-Voice/intents/blob/main/docs/slot_combinations.md

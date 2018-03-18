#!/usr/bin/env python3

"""
    Downloads haikus from different collections on the internet.
"""

import string
from requests_html import HTMLSession


# Redefine what 'alphabet' means so we don't filter out newlines and spaces.
ALPHABET = frozenset(string.ascii_lowercase + '\n' + ' ')


def preprocess(text, use_ascii=True):
    """
        Preprocess text. Converts to lowercase and filters non-alphabetic characters.
        Defaults to defining alphabetic characters as ascii-alphabetic
        Examples:
        >>> text = 'ABC.,#'
        >>> ''.join(preprocess(text))
        'abc'
        >>> text = 'ÈÆÖÉEAEOE,.%'
        >>> ''.join(preprocess(text, use_ascii=False))
        'èæöéeaeoe'
    """
    if use_ascii:
        return filter(ALPHABET.__contains__, text.lower())
    return filter(str.isalpha, text.lower())


def scrape_haikus():
    session = HTMLSession()
    dataset = []

    # I can't loop over all the URLs because most of them are entirely different.

    url = 'http://www.hsa-haiku.org/hendersonawards/henderson.htm'
    r = session.get(url)
    haikus = r.html.find('td > blockquote > p')
    haikus = (h.text for h in haikus)
    haikus = (''.join(preprocess(h)).strip() for h in haikus)
    haikus = filter(lambda x: len(x) > 1, haikus)
    haikus = (h.split('\n') for h in haikus)
    haikus = list('\t'.join(' '.join(line.split()) for line in h if len(line) > 1) for h in haikus)
    dataset += haikus

    print(f'Scraped {len(haikus)} haikus from {url}')

    url = 'http://www.hsa-haiku.org/bradyawards/brady.htm'
    r = session.get(url)
    haikus = r.html.find('td > blockquote > p')
    haikus = (h.text for h in haikus)
    haikus = (''.join(preprocess(h)).strip() for h in haikus)
    haikus = filter(lambda x: len(x) > 1, haikus)
    haikus = (h.split('\n') for h in haikus)
    haikus = list('\t'.join(' '.join(line.split()) for line in h if len(line) > 1) for h in haikus)
    dataset += haikus

    print(f'Scraped {len(haikus)} haikus from {url}')

    url = 'http://www.hsa-haiku.org/museumhaikuliteratureawards/museumhaikuliterature-award.htm'
    r = session.get(url)
    # Ignore the haikus in <td></td>s because it'd be too hard to parse out the author names et al.
    haikus = r.html.find('p.haiku')
    haikus = (h.text for h in haikus)
    haikus = (''.join(preprocess(h)).strip() for h in haikus)
    haikus = filter(lambda x: len(x) > 1, haikus)
    haikus = (h.split('\n') for h in haikus)
    haikus = list('\t'.join(' '.join(line.split()) for line in h if len(line) > 1) for h in haikus)
    dataset += haikus

    print(f'Scraped {len(haikus)} haikus from {url}')

    url = 'http://www.hsa-haiku.org/virgilioawards/virgilio.htm'
    r = session.get(url)
    # Not just <p></p>s...
    haikus = r.html.find('.haiku')
    haikus = (h.text for h in haikus)
    haikus = (''.join(preprocess(h)).strip() for h in haikus)
    haikus = filter(lambda x: len(x) > 1, haikus)
    haikus = (h.split('\n') for h in haikus)
    haikus = list('\t'.join(' '.join(line.split()) for line in h if len(line) > 1) for h in haikus)
    dataset += haikus

    print(f'Scraped {len(haikus)} haikus from {url}')

    url = 'https://www.thehaikufoundation.org/per-diem-archive/'
    r = session.get(url)
    urls = r.html.find('li > a')
    urls = (u.attrs['href'] for u in urls)
    urls = filter(lambda x: 'IDcat' in x, urls)
    urls = (f'https://www.thehaikufoundation.org{u}' for u in urls)

    all_haikus = []
    for url in urls:
        r = session.get(url)
        try:
            haikus = r.html.find('td > pre')
            haikus = (h.text for h in haikus)
            haikus = (''.join(preprocess(h)).strip() for h in haikus)
            haikus = filter(lambda x: len(x) > 1, haikus)
            haikus = (h.split('\n') for h in haikus)
            haikus = ('\t'.join(' '.join(line.split()) for line in h if len(line) > 1) for h in haikus)

            all_haikus += list(haikus)
        except:
            pass
    dataset += all_haikus

    print(f'Scraped {len(all_haikus)} haikus from {url}')

    url = 'https://www.ahapoetry.com/aadoh/h_dictionary.htm'
    r = session.get(url)
    urls = r.html.find('p > a')
    urls = (u.attrs['href'] for u in urls)
    urls = (f'https://www.ahapoetry.com/aadoh/{u}' for u in urls)

    def key(x):
        """Is a given x a haiku?"""
        try:
            return x.attrs['align'] == 'center'
        except:
            return False

    all_haikus = []
    for url in urls:
        r = session.get(url)
        haikus = r.html.find('p')
        haikus = filter(key, haikus)
        haikus = (h.text for h in haikus)
        haikus = (''.join(preprocess(h)).strip() for h in haikus)
        haikus = filter(lambda x: len(x) > 1, haikus)
        haikus = (h.split('\n') for h in haikus)
        haikus = ('\t'.join(' '.join(line.split()) for line in h if len(line) > 1) for h in haikus)

        all_haikus += list(haikus)

    dataset += all_haikus

    print(f'Scraped {len(all_haikus)} haikus from {url}')

    return dataset


def main():
    filename = 'haikus.txt'
    data = scrape_haikus()
    unique_data = set(data)
    print(f'Scraped a total of {len(data)} haikus ({len(data) - len(unique_data)}) duplicates')
    print(f'Saving haikus to {filename}')

    with open(filename, 'w') as f:
        for haiku in unique_data:
            f.write(haiku + '\n')


if __name__ == '__main__':
    main()

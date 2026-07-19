"use client";

import { FormEvent, useEffect, useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { getExploreBooks, getExploreGenres } from "@/services/BookService";
import type { Book } from "@/hooks/useBooks";

export default function ExplorePage() {
    const [books, setBooks] = useState<Book[]>([]);
    const [genres, setGenres] = useState<string[]>(["All"]);
    const [genre, setGenre] = useState<string>("All");
    const [searchField, setSearchField] = useState<"author" | "publisher" | "title">("title");
    const [query, setQuery] = useState<string>("");
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const filterBooksByTitle = (items: Book[], keyword: string) => {
        const trimmed = keyword.trim().toLowerCase();
        if (!trimmed) {
            return items;
        }

        return items.filter((book) => book.title.toLowerCase().includes(trimmed));
    };

    useEffect(() => {
        const loadInitialData = async () => {
            setIsLoading(true);
            setError(null);
            try {
                const [genreList, initialBooks] = await Promise.all([
                    getExploreGenres(),
                    getExploreBooks({ limit: 12 }),
                ]);
                setGenres(["All", ...genreList.filter(Boolean)]);
                setBooks(initialBooks);
            } catch (err) {
                setError(err instanceof Error ? err.message : "Failed to load explore books");
            } finally {
                setIsLoading(false);
            }
        };

        loadInitialData();
    }, []);

    const handleSearch = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setIsLoading(true);
        setError(null);
        try {
            const data = await getExploreBooks({
                limit: 12,
                genre,
                searchField: searchField,
                query,
            });
            setBooks(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to search books");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 dark:bg-slate-900">
            <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 sm:py-16 lg:px-8">
                <h1 className="mb-4 text-3xl font-bold text-slate-900 sm:text-4xl dark:text-white">Explore Books</h1>

                <form onSubmit={handleSearch} className="mb-8 space-y-5 rounded-3xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5 dark:border-slate-800 dark:bg-slate-950">
                    <p className="max-w-2xl text-sm text-slate-700 dark:text-slate-300">
                        The page loads a small book set first. Use the controls below, then click Search to fetch only the matching books.
                    </p>

                    <div className="grid gap-4 lg:grid-cols-[1fr_auto] lg:items-end">
                        <div className="space-y-3">
                            <p className="text-sm font-semibold text-slate-900 dark:text-white">Genre</p>
                            <div className="flex flex-wrap gap-3">
                                {genres.map((g) => (
                                    <label key={g} className="inline-flex items-center gap-2 rounded-full border border-slate-200 px-3 py-2 text-sm dark:border-slate-800">
                                        <input
                                            type="radio"
                                            name="genre"
                                            value={g}
                                            checked={genre === g}
                                            onChange={() => setGenre(g)}
                                            className="h-4 w-4"
                                        />
                                        <span>{g}</span>
                                    </label>
                                ))}
                            </div>
                        </div>

                        <div className="grid gap-3 sm:grid-cols-[160px_1fr]">
                            <label className="space-y-2">
                                <span className="text-sm font-semibold text-slate-900 dark:text-white">Search by</span>
                                <select
                                    aria-label="Search field"
                                    value={searchField}
                                    onChange={(e) => setSearchField(e.target.value as "author" | "publisher" | "title")}
                                    className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm dark:border-slate-800 dark:bg-slate-900"
                                >
                                    <option value="author">Author</option>
                                    <option value="publisher">Publisher</option>
                                    <option value="title">Title</option>
                                </select>
                            </label>

                            <label className="space-y-2">
                                <span className="text-sm font-semibold text-slate-900 dark:text-white">Keyword</span>
                                <input
                                    placeholder={`Search ${searchField}...`}
                                    value={query}
                                    onChange={(e) => setQuery(e.target.value)}
                                    className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm dark:border-slate-800 dark:bg-slate-900"
                                />
                            </label>
                        </div>
                    </div>

                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                        <button
                            type="submit"
                            className="rounded-full bg-slate-950 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-slate-800 sm:w-auto dark:bg-white dark:text-slate-950 dark:hover:bg-slate-200"
                        >
                            Search
                        </button>
                        <button
                            type="button"
                            onClick={async () => {
                                setGenre("All");
                                setSearchField("title");
                                setQuery("");
                                setIsLoading(true);
                                setError(null);
                                try {
                                    const data = await getExploreBooks({ limit: 12 });
                                    setBooks(data);
                                } catch (err) {
                                    setError(err instanceof Error ? err.message : "Failed to reset books");
                                } finally {
                                    setIsLoading(false);
                                }
                            }}
                            className="rounded-full border border-slate-200 px-5 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-100 sm:w-auto dark:border-slate-800 dark:text-slate-300 dark:hover:bg-slate-900"
                        >
                            Reset
                        </button>
                    </div>
                </form>

                {isLoading ? (
                    <div className="py-12 text-center text-slate-600">Loading books...</div>
                ) : error ? (
                    <div className="py-12 text-center text-red-500">{error}</div>
                ) : (
                    <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
                        {books.map((book) => (
                            <Link
                                key={book.id}
                                href={`/books/${book.id}`}
                                className="group overflow-hidden rounded-4xl border border-slate-200 bg-white shadow-sm transition hover:-translate-y-1 dark:border-slate-800 dark:bg-slate-950"
                            >
                                <div className="relative aspect-3/4 w-full bg-slate-100 dark:bg-slate-900">
                                    {book.image_url ? (
                                        <Image src={book.image_url} alt={book.title} fill className="object-cover" />
                                    ) : null}
                                </div>
                                <div className="space-y-1 p-4">
                                    <h2 className="text-lg font-semibold text-slate-900 dark:text-white">{book.title}</h2>
                                    <p className="text-sm text-slate-600 dark:text-slate-300">{book.author}</p>
                                    <p className="text-xs text-slate-500">{book.publisher} • {(book.genres?.length ? book.genres.join(", ") : book.category) ?? "Unknown"}</p>
                                </div>
                            </Link>
                        ))}

                        {books.length === 0 ? (
                            <div className="rounded-3xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500 dark:border-slate-700 dark:text-slate-400">
                                No books found.
                            </div>
                        ) : null}
                    </div>
                )}
            </div>
        </div>
    );
}
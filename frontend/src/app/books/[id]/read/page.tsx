"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { RequireAuth } from "@/components/AuthGuard";
import BookImageReader from "@/components/BookImageReader";
import { getBook } from "@/services/BookService";
import type { Book } from "@/hooks/useBooks";

function ReaderContent() {
	const params = useParams();
	const bookId = Number(Array.isArray(params.id) ? params.id[0] : params.id);
	const [book, setBook] = useState<Book | null>(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		const loadBook = async () => {
			setLoading(true);
			setError(null);
			try {
				const data = await getBook(bookId);
				setBook(data);
			} catch (err) {
				setError(err instanceof Error ? err.message : "Failed to load reader");
			} finally {
				setLoading(false);
			}
		};

		if (!Number.isNaN(bookId)) {
			loadBook();
		}
	}, [bookId]);

	if (loading) {
		return <div className="py-16 text-center text-sm text-slate-500">Loading reader...</div>;
	}

	if (error || !book) {
		return <div className="py-16 text-center text-sm text-red-500">{error ?? "Book not found"}</div>;
	}

	if (!book.content_file) {
		return (
			<div className="min-h-[70vh] rounded-4xl border border-slate-200 bg-white p-6 shadow-2xl shadow-slate-950/5 dark:border-slate-800 dark:bg-slate-950 sm:p-8">
				<div className="mb-8 flex items-center justify-between gap-4">
					<div>
						<p className="text-sm uppercase tracking-[0.3em] text-slate-500">Reader</p>
						<h1 className="mt-2 text-3xl font-black tracking-tight text-slate-950 dark:text-white">{book.title}</h1>
					</div>
					<Link href={`/books/${book.id}`} className="rounded-full border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-950 hover:text-slate-950 dark:border-slate-700 dark:text-slate-300 dark:hover:border-white dark:hover:text-white">
						Back
					</Link>
				</div>
				<div className="py-8 text-sm text-slate-600 dark:text-slate-300">This book does not have image content yet.</div>
			</div>
		);
	}

	return (
		<div className="min-h-[70vh] rounded-4xl border border-slate-200 bg-white p-6 shadow-2xl shadow-slate-950/5 dark:border-slate-800 dark:bg-slate-950 sm:p-8">
			<div className="mb-8 flex items-center justify-between gap-4">
				<div>
					<p className="text-sm uppercase tracking-[0.3em] text-slate-500">Reader</p>
					<h1 className="mt-2 text-3xl font-black tracking-tight text-slate-950 dark:text-white">{book.title}</h1>
					<p className="mt-2 text-sm text-slate-500">By {book.author}</p>
				</div>
				<Link href={`/books/${book.id}`} className="rounded-full border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-950 hover:text-slate-950 dark:border-slate-700 dark:text-slate-300 dark:hover:border-white dark:hover:text-white">
					Back
				</Link>
			</div>

			<BookImageReader bookId={book.id} />
		</div>
	);
}

export default function BookReadPage() {
	return (
		<RequireAuth>
			<ReaderContent />
		</RequireAuth>
	);
}

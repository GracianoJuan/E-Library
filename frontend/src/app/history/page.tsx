"use client";

import Link from "next/link";
import Image from "next/image";
import { useEffect, useState } from "react";
import { RequireAuth } from "@/components/AuthGuard";
import { deleteHistory, getHistory, type HistoryItem } from "@/services/HistoryService";

function HistoryContent() {
	const [history, setHistory] = useState<HistoryItem[]>([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [deletingBookId, setDeletingBookId] = useState<number | null>(null);

	useEffect(() => {
		const loadHistory = async () => {
			setLoading(true);
			setError(null);
			try {
				const data = await getHistory();
				setHistory(data);
			} catch (err) {
				setError(err instanceof Error ? err.message : "Failed to load history");
			} finally {
				setLoading(false);
			}
		};

		loadHistory();
	}, []);

	const handleDelete = async (bookId: number) => {
		setDeletingBookId(bookId);
		setError(null);
		try {
			await deleteHistory(bookId);
			setHistory((current) => current.filter((item) => item.book.id !== bookId));
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to delete history");
		} finally {
			setDeletingBookId(null);
		}
	};

	if (loading) {
		return <div className="py-16 text-center text-sm text-slate-500">Loading history...</div>;
	}

	if (error) {
		return <div className="py-16 text-center text-sm text-red-500">{error}</div>;
	}

	return (
		<div className="space-y-6">
			<div>
				<p className="text-sm uppercase tracking-[0.3em] text-slate-500">History</p>
				<h1 className="mt-3 text-3xl font-black tracking-tight text-slate-950 sm:text-4xl dark:text-white">Reading History</h1>
			</div>

			<div className="grid gap-4 sm:grid-cols-4 xl:grid-cols-5">
				{history.map((item) => (
					<div
						key={item.id}
						className="group overflow-hidden rounded-4xl border border-slate-200 bg-white shadow-lg shadow-slate-950/5 transition hover:-translate-y-1 hover:shadow-2xl dark:border-slate-800 dark:bg-slate-950"
					>
						<Link href={`/books/${item.book.id}`}>
							<div className="relative aspect-3/4 w-full bg-slate-100 dark:bg-slate-900">
							<Image src={item.book.image_url} alt={item.book.title} fill className="object-cover" />
						</div>
							<div className="space-y-2 p-5">
							<p className="text-xs uppercase tracking-[0.2em] text-slate-500">Read at</p>
							<p className="text-sm text-slate-600 dark:text-slate-300">{new Date(item.read_at).toLocaleString()}</p>
							<h2 className="text-xl font-bold text-slate-950 dark:text-white">{item.book.title}</h2>
							<p className="text-sm text-slate-500">{item.book.author}</p>
						</div>
						</Link>
						<div className="px-5 pb-5">
							<button
								type="button"
								onClick={() => handleDelete(item.book.id)}
								disabled={deletingBookId === item.book.id}
								className="w-full rounded-2xl border border-rose-200 px-4 py-2.5 text-sm font-semibold text-rose-600 transition hover:border-rose-400 hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-rose-900/60 dark:text-rose-300 dark:hover:bg-rose-950/40"
							>
								{deletingBookId === item.book.id ? "Deleting..." : "Delete from history"}
							</button>
						</div>
					</div>
				))}
			</div>

			{history.length === 0 ? (
				<div className="rounded-3xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500 dark:border-slate-700 dark:text-slate-400">
					No reading history yet.
				</div>
			) : null}
		</div>
	);
}

export default function HistoryPage() {
	return (
		<RequireAuth>
			<HistoryContent />
		</RequireAuth>
	);
}
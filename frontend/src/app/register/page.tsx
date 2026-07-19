"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { PublicOnly } from "@/components/AuthGuard";
import { useAuth } from "@/hooks/useAuth";

function RegisterForm() {
	const router = useRouter();
	const { register, isLoading } = useAuth();
	const [name, setName] = useState("");
	const [email, setEmail] = useState("");
	const [password, setPassword] = useState("");
	const [error, setError] = useState<string | null>(null);
	const [submitting, setSubmitting] = useState(false);

	const handleSubmit = async (event: React.FormEvent) => {
		event.preventDefault();
		setSubmitting(true);
		setError(null);

		try {
			await register({ name, email, password });
			router.replace("/login");
		} catch (err) {
			setError(err instanceof Error ? err.message : "Registration failed");
		} finally {
			setSubmitting(false);
		}
	};

	return (
		<div className="mx-auto flex min-h-[70vh] w-full max-w-md items-center px-4 py-8 sm:px-6 lg:px-8">
			<div className="w-full rounded-4xl border border-slate-200 bg-white p-6 shadow-2xl shadow-slate-950/5 sm:p-8 dark:border-slate-800 dark:bg-slate-950">
				<div className="mb-8 text-center">
					<p className="text-sm uppercase tracking-[0.3em] text-slate-500">Create account</p>
					<h1 className="mt-3 text-3xl font-black tracking-tight text-slate-950 dark:text-white">Register</h1>
					<p className="mt-2 text-sm text-slate-500 dark:text-slate-400">Set up your account to save history and get recommendations.</p>
				</div>

				<form onSubmit={handleSubmit} className="space-y-5">
					<label className="block">
						<span className="mb-2 block text-sm font-medium text-slate-700 dark:text-slate-300">Name</span>
						<input
							type="text"
							value={name}
							onChange={(event) => setName(event.target.value)}
							required
							className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-slate-950 outline-none transition focus:border-slate-950 dark:border-slate-700 dark:bg-slate-900 dark:text-white dark:focus:border-white"
							placeholder="Your name"
						/>
					</label>

					<label className="block">
						<span className="mb-2 block text-sm font-medium text-slate-700 dark:text-slate-300">Email</span>
						<input
							type="email"
							value={email}
							onChange={(event) => setEmail(event.target.value)}
							required
							className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-slate-950 outline-none transition focus:border-slate-950 dark:border-slate-700 dark:bg-slate-900 dark:text-white dark:focus:border-white"
							placeholder="you@example.com"
						/>
					</label>

					<label className="block">
						<span className="mb-2 block text-sm font-medium text-slate-700 dark:text-slate-300">Password</span>
						<input
							type="password"
							value={password}
							onChange={(event) => setPassword(event.target.value)}
							required
							className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-slate-950 outline-none transition focus:border-slate-950 dark:border-slate-700 dark:bg-slate-900 dark:text-white dark:focus:border-white"
							placeholder="••••••••"
						/>
					</label>

					{error ? <p className="text-sm text-red-500">{error}</p> : null}

					<button
						type="submit"
						disabled={submitting || isLoading}
						className="w-full rounded-2xl bg-slate-950 px-4 py-3 font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-white dark:text-slate-950 dark:hover:bg-slate-200"
					>
						{submitting ? "Creating account..." : "Register"}
					</button>
				</form>

				<p className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
					Already have an account? <Link href="/login" className="font-semibold text-slate-950 dark:text-white">Login</Link>
				</p>
			</div>
		</div>
	);
}

export default function RegisterPage() {
	return (
		<PublicOnly>
			<RegisterForm />
		</PublicOnly>
	);
}

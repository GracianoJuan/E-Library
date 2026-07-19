const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type RequestOptions = {
	method?: string;
	body?: unknown;
	withAuth?: boolean;
};

function getAuthToken(): string | null {
	if (typeof window === "undefined") return null;
	return localStorage.getItem("access_token");
}

export async function apiRequest<T>(
	path: string,
	options: RequestOptions = {}
): Promise<T> {
	const method = options.method ?? "GET";
	const headers: HeadersInit = {
		"Content-Type": "application/json",
	};

	// Add authorization header if token exists and withAuth is not false
	const token = getAuthToken();
	if (options.withAuth !== false && token) {
		headers["Authorization"] = `Bearer ${token}`;
	}

	const response = await fetch(`${API_BASE_URL}${path}`, {
		method,
		headers,
		credentials: "include",
		body: options.body ? JSON.stringify(options.body) : undefined,
	});

	if (!response.ok) {
		let message = "Request failed";
		try {
			const errorBody = await response.json();
			message = errorBody.detail ?? message;
		} catch {
			// Keep default message when body is not json.
		}
		throw new Error(message);
	}

	if (response.status === 204) {
		return undefined as T;
	}

	return (await response.json()) as T;
}

export function clearCsrfTokenCache() {
	const csrfTokenCache = null;
}
